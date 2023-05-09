"""
This script implements class detections which conatins the obj detections and has some functionality over

these detections 

"""
from OpenPCDet.tools.overlap import get_gt_poses, cvt_frame
import numpy as np
import os
import pickle as pk

class Detections():
    def __init__(self, detection_path, transform) -> None:
        np.set_printoptions(suppress=True)

        assert os.path.isfile(detection_path), 'Detection pkl file does not exist'
        self.gt_poses = get_gt_poses("/media/sandhu/SSD/dataset/kitti/odometry/dataset", "08")
        self.transform = transform
        self.detection_path = detection_path
        self.detections = self.loading_detections()
        self.lidar_path = list(self.detections.keys())
        

    def loading_detections(self):
        assert os.path.splitext(self.detection_path)[1] == '.pkl', 'File is not a .pkl file'
        
        with open(self.detection_path, "rb") as f:
            return pk.load(f)

    def get_frames(self):
        self.scan_base_path = os.path.dirname(next(iter(self.detections)))
        self.frames = [os.path.split(i)[1] for i in list(self.detections.keys())]

    
    def __getitem__(self, idx):
        key = self.lidar_path[idx]
        frame_number = int(os.path.splitext(os.path.basename(key))[0])
        
        boxes = self.detections[key]['boxes']
        labels = self.detections[key]['labels'].reshape(-1, 1)

        frame_col = np.full((len(labels), 1), frame_number)
        if self.transform:
            boxes = cvt_frame(self.gt_poses, idx, boxes)
        assert len(boxes) == len(labels), "Bounding box does not match labels"
        # return np.hstack((frame_col, boxes, labels))
        return key, boxes, labels

    def __len__(self):
        return len(self.lidar_path)


if __name__ == "__main__":
    path = '/home/sandhu/project/p04-masterproject/tracked_det.pkl'
    det = Detections(path, True)

    transformed_det = {}
    for idx in range(len(det)):
        key, box, lab = det[idx]
        transformed_det[key] = {'boxes': box, 'labels': lab}

    with open("transformed_bb_det.pkl", 'wb') as f:
        pk.dump(transformed_det, f) 
        print("file dumped")
        
