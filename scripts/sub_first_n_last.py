import sys
import numpy as np
import os
import math
import utils
from tqdm import tqdm




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python first_n_last.py [tracking_file_path] [yes/no to transform boxes or not]")
        sys.exit(1)
    
    data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
    sequence = "08"
    gt_poses = utils.get_gt_poses(data_path, sequence)

    tracked_dataset_path = sys.argv[1]    
    tracked_files_path = [os.path.join(tracked_dataset_path, file) for file in os.listdir(tracked_dataset_path)]
    
    lidar_files = list(utils.load_pickle("obj_boxes.pkl").keys())

    dataset = [np.loadtxt(file_path) for file_path in tracked_files_path]
    dataset = np.concatenate(dataset) 
    dataset = dataset[dataset[:, 0].argsort()]
    frames = np.unique(dataset[:, 0])
    
