import os, time
import numpy as np
import open3d as o3d
import torch,random
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device=torch.device("cuda:0")

def build_dense(xyz, rgb, mins_glb, dims, device, segment=None):

    idx = np.floor(xyz - mins_glb).astype(np.int64)
    idx = torch.from_numpy(idx).to(device)
    rgb = torch.from_numpy(rgb).to(device)

    W, H, D = dims
    dense_sum = torch.zeros((W, H, D, 3), dtype=torch.float64, device=device)
    dense_cnt = torch.zeros((W, H, D), dtype=torch.float64, device=device)

    # 展平坐标索引到 1D 索引，方便 scatter 操作
    linear_idx = idx[:, 0] * (H * D) + idx[:, 1] * D + idx[:, 2]

    # 对 RGB 三个通道分别 scatter_add
    for c in range(3):
        dense_sum[..., c].view(-1).scatter_add_(
            0, linear_idx, rgb[:, c].to(torch.float64)
        )

    # 对计数 scatter_add
    dense_cnt.view(-1).scatter_add_(
        0, linear_idx, torch.ones_like(linear_idx, dtype=torch.float64)
    )

    # 计算平均颜色
    dense_rgb = torch.zeros((W, H, D, 3), dtype=torch.float64, device=device)
    dense_rgb[dense_cnt > 0] = dense_sum[dense_cnt > 0] / dense_cnt[dense_cnt > 0].unsqueeze(-1)

    # 构造最终 dense
    dense = torch.zeros((W, H, D, 5), dtype=torch.float64, device=device)
    dense[..., 0] = (dense_cnt > 0).float()              # occupancy
    dense[..., 1:4] = dense_rgb                          # averaged RGB
    if segment is not None:
        segment = torch.as_tensor(segment, dtype=torch.float64, device=device).squeeze(-1)
        dense[idx[:, 0], idx[:, 1], idx[:, 2], 4] = segment
    else:
        dense[..., 4] = 0.0
    return dense


def reconstruct(dense_rec, mins_glb,dense_B):
    occ_mask   = dense_B[...,0]==1
    #print(occ_mask.sum())
    coords_ijk = torch.nonzero(occ_mask).cpu().numpy()
    #print(coords_ijk.shape)
    xyz        = coords_ijk.astype(np.float32) 
    #print(dense_rec.shape)
    colors = dense_rec[occ_mask]  # 直接索引被占用的体素颜色
    colors = colors.clip(0, 1).cpu().numpy()
    segment = dense_B[..., 4:5][occ_mask].cpu().numpy()
    return xyz,colors,segment
    """ pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"[INFO] {title}: {len(xyz)} points")
    o3d.visualization.draw_geometries([pcd], window_name=title) """
def estimate_normals_o3d(xyz: np.ndarray):
    
    device = o3d.core.Device("CUDA:0")
    tensor_xyz = o3d.core.Tensor(xyz, dtype=o3d.core.float32, device=device)
    pcd = o3d.t.geometry.PointCloud(tensor_xyz)
    pcd.estimate_normals( max_nn=30,radius=0.05,)
    normals = np.asarray(pcd.point["normals"].cpu().numpy())
    return normals

def low_freq_mutate_3d_torch(amp_src, amp_trg, L=0.1):
    """
    用 PyTorch 实现三维傅里叶振幅谱的低频区域替换。
    
    参数:
        amp_src: torch.Tensor, [B, D, H, W, C]
        amp_trg: torch.Tensor, [B, D, H, W, C]
        L: float, 低频区域比例 (0 < L < 0.5)
    
    返回:
        torch.Tensor, 替换低频后的源振幅谱 (同形状)
    """
    amp_src.unsqueeze_(0)
    amp_trg.unsqueeze_(0)

    B, D, H, W, C = amp_src.shape

    # 频谱中心化
    a_src = torch.fft.fftshift(amp_src, dim=(1, 2, 3))
    a_trg = torch.fft.fftshift(amp_trg, dim=(1, 2, 3))

    # 低频块半径
    b = int(torch.floor(torch.tensor(min(D, H, W) * L)).item())

    # 中心坐标
    c_d, c_h, c_w = D // 2, H // 2, W // 2

    # 替换范围
    d1, d2 = c_d - b, c_d + b + 1
    h1, h2 = c_h - b, c_h + b + 1
    w1, w2 = c_w - b, c_w + b + 1

    # 替换低频立方体区域
    a_src[:, d1:d2, h1:h2, w1:w2, :] = a_trg[:, d1:d2, h1:h2, w1:w2, :]

    a_src = torch.fft.ifftshift(a_src, dim=(1, 2, 3))
    return a_src

def mix(source,target,save):
    path_B = source  #B是target
    path_A = target
    """ if os.path.exists(path_A):
        print("文件存在！")
    else:
        print("文件不存在！") """
    coordA = np.load(os.path.join(path_A, 'coord.npy'))
    coordB = np.load(os.path.join(path_B, 'coord.npy'))
    colorA = np.load(os.path.join(path_A, 'color.npy'))/255  #把颜色变成0-1之间
    colorB = np.load(os.path.join(path_B, 'color.npy'))/255
    segment=np.load(os.path.join(path_B, 'segment.npy'))
    print(segment.shape,coordB.shape)
    coordA = ((coordA - coordA.min(axis=0)) * 30).astype(np.int64)
    coordB = ((coordB - coordB.min(axis=0)) * 30).astype(np.int64)

    #mix_coord,mix_segment=dedup_with_segment(torch.tensor(coordB),torch.tensor(segment),torch.tensor(coordB_))  #由于上面*30，保留去重后的segment标签

    mins_glb = np.minimum(coordA.min(axis=0), coordB.min(axis=0)).astype(np.float32)
    maxs_glb = np.maximum(coordA.max(axis=0), coordB.max(axis=0)).astype(np.float32)
    dims     = (np.ceil((maxs_glb - mins_glb) ).astype(int) + 1)
    W, H, D  = map(int, dims)

    dense_A = build_dense(coordA, colorA, mins_glb, dims, device,segment=None)
    dense_B = build_dense(coordB, colorB, mins_glb, dims, device,segment)

    F_A = torch.fft.fftn(dense_A[..., 1:4], dim=(0,1,2))
    F_B = torch.fft.fftn(dense_B[..., 1:4], dim=(0,1,2))


    amp_A   = torch.abs(F_A)
    amp_B   = torch.abs(F_B)
    phase_B = torch.angle(F_B)

    amp_B = low_freq_mutate_3d_torch(amp_B, amp_A, 0.5).squeeze(0) #用A的低频替换B的低频，替换太多颜色很杂

    F_mix = amp_B * torch.exp(1j * phase_B)
    
    dense_mix = torch.fft.ifftn(F_mix, s=(W, H, D), dim=(0, 1, 2)).real
    #print(torch.allclose(dense_mix, dense_B[...,1:4], atol=1e-8, rtol=1e-5))
    mix_coord,mix_color,mix_segment=reconstruct(dense_mix, mins_glb, dense_B)
    #print("done")
    mix_color=mix_color*255
    #print(mix_coord.shape)
    mix_normal=estimate_normals_o3d(mix_coord)
    print(mix_color.shape,mix_coord.shape,mix_normal.shape,mix_segment.shape)
    if save is not None:
        os.makedirs(save, exist_ok=True)
        np.save(os.path.join(save, "coord.npy"), mix_coord)
        np.save(os.path.join(save, "color.npy"), mix_color)
        np.save(os.path.join(save, "normal.npy"), mix_normal)
        np.save(os.path.join(save, "segment.npy"), mix_segment)

def work(source_root, target_root, mix_scene_func, save_root=None, seed=42):
    """
    遍历 source/train 下的所有场景，每个场景随机匹配一个 target/train 下的场景并混合

    Args:
        source_root (str): 源域 train 文件夹路径，如 '/A432/lhy/Dataset/source/train'
        target_root (str): 目标域 train 文件夹路径，如 '/A432/lhy/Dataset/target/train'
        mix_scene_func (callable): 混合函数 f(source_scene_path, target_scene_path, save_path)
        save_root (str): 可选，混合后结果保存目录，若为 None 则不保存
        seed (int): 随机种子保证可复现
    """
    random.seed(seed)

    # 获取所有场景目录
    source_scenes = sorted([d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))])
    target_scenes = sorted([d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d))])

    print(f"[INFO] Source scenes: {len(source_scenes)}, Target scenes: {len(target_scenes)}")

    for scene in source_scenes:
        src_path = os.path.join(source_root, scene)
        tgt_scene = random.choice(target_scenes)
        tgt_path = os.path.join(target_root, tgt_scene)

        print(f"→ Mixing {scene}  ×  {tgt_scene}")

        # 保存路径
        save_path = None
        if save_root is not None:
            os.makedirs(save_root, exist_ok=True)
            save_path = os.path.join(save_root, f"{scene}_mix_{tgt_scene}")
            os.makedirs(save_path, exist_ok=True)

        # 调用用户定义的混合函数
        mix(src_path, tgt_path, save_path)

    print("[DONE] 全部混合完成 ")

work(
    source_root="/A432/lhy/Dataset/matterport3d/train_pro",       # 源域数据集 train 文件夹
    target_root="/A432/lhy/Dataset/structured3d/train_pro",     # 目标域数据集 train 文件夹
    mix_scene_func=mix,                                    
    save_root="/A432/lhy/Dataset/matterport2structure/train",  # 混合后结果保存目录
    seed=42,                                              
)
""" source='/A432/lhy/Dataset/structured3d/test/scene_00001_room_906322'
target='/A432/lhy/Dataset/matterport3d/test/2t7WUuJeko7_00'
mix(source,target,"/A432/lhy/Dataset/0.5") """
#python Fourier/compare.py
