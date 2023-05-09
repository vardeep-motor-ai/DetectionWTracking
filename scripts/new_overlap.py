import pickle
from numpy.core.overrides import set_module 
import torch
import numpy as np
from pcdet.utils import common_utils
import math

import open3d as o3d
import pykitti
import os
import matplotlib.pyplot as plt
from visual_utils import visualize_utils

class Dataset():
    """
    This dataset class consists of all information of the dataset i.e. where are all the scans
    and annos and the transformation matrices
    """

    def __init__(self, dataset_path, sequence, detection_path):
        self.dataset_path = dataset_path
        self.sequence = sequence
        self.detection_path = detection_path
        self.lidar_scans = None
        self.annos = None
        self._get_gt_poses()
        self._get_scan_n_anno_path()

    def _get_gt_poses(self):
        
        """
        This function loads the GT poses on the Velodyne coordinate frame.
        """
        
        data = pykitti.odometry(self.dataset_path, self.sequence)
        T_cam_velo = data.calib.T_cam0_velo
        T_velo_cam = np.linalg.inv(T_cam_velo)
        self.gt_poses = T_velo_cam @ data.poses @ T_cam_velo
        

    def _get_scan_n_anno_path(self):
        scan_path = os.path.join(self.dataset_path, "sequences", self.sequence, "velodyne")
        anno_path = os.path.join(self.dataset_path, "sequences", self.sequence, "labels")
        # self.lidar_scans = [os.path.join(root, file) for root, _, file in os.walk(scan_path)].sort()
        # self.annos = [os.path.join(root, file) for root, _, files in os.walk(anno_path)].sort()
        self.lidar_scans = [os.path.join(scan_path, file) for file in os.listdir(scan_path)]
        self.annos = [os.path.join(anno_path, file) for file in os.listdir(anno_path)]
        assert len(self.lidar_scans) == len(self.annos)


    def __getitem__(self, index):
        return self.lidar_scans[index]

    def __len__(self):
        return len(self.lidar_scans)
        
dataset_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
sequence = "08"
detection_path = "obj_boxes.pkl"
obj = Dataset(dataset_path, sequence, detection_path)
print(obj[0])