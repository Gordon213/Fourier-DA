import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
show_seg=False
#root为文件夹路径
root=r"D:\vscodefiles\python\2azQ1b91cZZ_03_mix_scene_00196_room_10116"
dir=['color.npy','coord.npy','segment.npy']
color=np.load(root+'\\'+dir[0])
xyz=np.load(root+'\\'+dir[1])
seg=np.load(root+'\\'+ dir[2])

np.random.seed(42)
n_points = seg.shape[0]

# 创建颜色映射
unique_segments = np.unique(seg)
print(unique_segments)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_segments)))[:, :3]  # 只取RGB，不要alpha
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# 为每个点分配颜色
point_colors = np.zeros((n_points, 3))
for i, seg_id in enumerate(seg):
    seg_index = np.where(unique_segments == seg_id)[0][0]
    point_colors[i] = colors[seg_index]
if show_seg==False:
    pcd.colors = o3d.utility.Vector3dVector(color/255)
else:
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
# 可视化
o3d.visualization.draw_geometries([pcd])
