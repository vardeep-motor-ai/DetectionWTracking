import numpy as np
import os
import utils
import math

def bb_to_global(points, T_0_current):
    
    xyz = utils.points_to_homo(points[: , :3])
    
    points_w = T_0_current @ xyz.T

    points_w = np.delete(points_w.T, -1, axis=1)
    heading = np.copy(points[:, -1])

    R = (T_0_current)[:3, :3]
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular :
        z = math.atan2(R[1,0], R[0,0])
    else :
        z = 0
    return np.hstack((points_w, points[:, 3:-1], (heading + z).reshape(-1,1)))


def main():
    path = "AB3DMOT/results/2022_01_26_13_28"
    data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
    sequence = "08"
    gt_poses = utils.get_gt_poses(data_path, sequence)


    file_paths = [os.path.join(path, file)for file in os.listdir(path)]
    all_files = [np.loadtxt(file_path) for file_path in file_paths]
    dataset = np.concatenate(all_files) 
    dataset = dataset[dataset[:, 0].argsort()]
    frames = np.unique(dataset[:, 0])
    new_dataset= []
    for frame in frames:
        frame_data = dataset[dataset[:, 0] == frame]
        bb = utils.bb_order_xyz(frame_data)
        bb_new = bb_to_global(bb, gt_poses[int(frame)])
        frame_data[:, -8:-1] = bb_new
        new_dataset.append(frame_data)

    new_dataset = np.concatenate(new_dataset)
    np.savetxt("bb_locally_tracked_saved_globally.txt", new_dataset)


main()