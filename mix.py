import os, time
import numpy as np
import open3d as o3d
import torch,random
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device=torch.device("cuda:0")

def build_dense(xyz, rgb, mins_glb, dims, device):
    idx = np.floor(xyz - mins_glb).astype(np.int64)  
    idx = torch.from_numpy(idx).to(device)
    rgb = torch.from_numpy(rgb).to(device)

    W, H, D = dims
    dense = torch.zeros((W, H, D, 4), dtype=torch.float64, device=device)

    dense[idx[:, 0], idx[:, 1], idx[:, 2], 0] = 1.0
    #print(dense[...,0].sum())
    dense[idx[:, 0], idx[:, 1], idx[:, 2], 1:] = rgb
    return dense

def reconstruct(dense_rec, mins_glb,dense_B):
    occ_mask   = dense_B[...,0]==1
    #print(occ_mask.sum())
    coords_ijk = torch.nonzero(occ_mask).cpu().numpy()
    #print(coords_ijk.shape)
    xyz        = coords_ijk.astype(np.float32) 
    #print(dense_rec.shape)
    colors = dense_rec[..., :][occ_mask]  # 直接索引被占用的体素颜色
    colors = colors.clip(0, 1).cpu().numpy()
    return xyz,colors
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

def dedup_with_segment(coords: torch.Tensor, segments: torch.Tensor):
    #去重点云坐标，并保留第一个出现的 segment 标签
    device = coords.device
    segments = segments.to(device)

    unique_coords, inverse_indices, counts = torch.unique(coords, dim=0, return_inverse=True, return_counts=True)

    order = torch.argsort(inverse_indices)
    inv_sorted = inverse_indices[order]
    first_mask = torch.ones_like(inv_sorted, dtype=torch.bool)
    first_mask[1:] = inv_sorted[1:] != inv_sorted[:-1]
    first_idx = order[first_mask]

    unique_segments = segments[first_idx]
    return unique_segments
    
import torch

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

    # 反中心化
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

    coordA = ((coordA - coordA.min(axis=0)) * 30).astype(np.int64)
    coordB = ((coordB - coordB.min(axis=0)) * 30).astype(np.int64)

    mix_segment=dedup_with_segment(torch.tensor(coordB),torch.tensor(segment))  #由于上面*30，保留去重后的segment标签

    mins_glb = np.minimum(coordA.min(axis=0), coordB.min(axis=0)).astype(np.float32)
    maxs_glb = np.maximum(coordA.max(axis=0), coordB.max(axis=0)).astype(np.float32)
    dims     = (np.ceil((maxs_glb - mins_glb) ).astype(int) + 1)
    W, H, D  = map(int, dims)

    dense_A = build_dense(coordA, colorA, mins_glb, dims, device)
    dense_B = build_dense(coordB, colorB, mins_glb, dims, device)
    #print(dense_A.dtype,dense_B.dtype)
    F_A = torch.fft.fftn(dense_A[..., 1:4], dim=(0,1,2))
    F_B = torch.fft.fftn(dense_B[..., 1:4], dim=(0,1,2))


    amp_A   = torch.abs(F_A)
    amp_B   = torch.abs(F_B)
    phase_B = torch.angle(F_B)
    #print(amp_B.shape)
    #print(amp_A.dtype)
    amp_B = low_freq_mutate_3d_torch(amp_B, amp_A, 0.1).squeeze(0)   #用A的低频替换B的低频，替换太多颜色很杂
    #print(amp_A.dtype)
    F_mix = amp_B * torch.exp(1j * phase_B)
    #print(F_mix.dtype,F_mix.shape)
    dense_mix = torch.fft.ifftn(F_mix, s=(W, H, D), dim=(0, 1, 2)).real
    #print(dense_mix.shape)
    mix_coord,mix_color=reconstruct(dense_mix, mins_glb, dense_B)
    #print("done")
    mix_color=mix_color*255
    mix_normal=estimate_normals_o3d(mix_coord)

    if save is not None:
        os.makedirs(save, exist_ok=True)
        np.save(os.path.join(save, "coord.npy"), mix_coord)
        np.save(os.path.join(save, "color.npy"), mix_color)
        np.save(os.path.join(save, "normal.npy"), mix_normal)
        np.save(os.path.join(save, "segment.npy"), mix_segment.numpy())
    """ print(unique_coords.shape,mix_coord.shape)
    unique_coords=unique_coords.cpu().numpy()
    print((unique_coords==mix_coord).all()) """

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
mix(source,target,"/A432/lhy/Dataset/0") """
#python Fourier/compare.py
