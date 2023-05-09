import open3d as o3d
import numpy as np
import meshplot as mp
from IPython.display import display
import utils
# !pip install meshplot

pcd = o3d.io.read_point_cloud("/home/sandhu/project/p04-masterproject/global_ply/004001.ply")
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
pallet = np.array([[253, 231, 37], [68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121]]) /255.0
# pallet = np.array([[253, 231, 37], [68, 1, 84], [72, 40, 120], [62, 73, 137], [49, 104, 142]]) /255.0

for i in range(len(colors)):
    # mean = np.mean(pcd.colors[i])
    # mean_pallet = np.mean(pallet, axis=1)
    print(pcd.colors[i])
    mean_full = np.full((len(pallet), 3), pcd.colors[i])
    print(mean_full)
    break
    dis = np.linalg.norm(pallet - mean_full)
    pcd.colors[i] = pallet[dis.argmin()]

    # pcd.colors[i] = pallet[np.random.randint(0, len(pallet))]

# pcd.paint_uniform_color([0, 0, 0])

utils.custom_visualization([pcd], "hello")