import numpy as np
import os
import utils
import math


def convert_bb_frame(points, Trans_mat, global_or_local: str):
    
    """ 
    This trans_mat converts points from current to 0 frame i.e. local to global frame
    arg global_or_local asks if you want to convert the boxes from local to global frame
    or from global to local. 
    Here we are converting from local to global
    """

    if global_or_local == "global":
        pass
    elif global_or_local == "local":
        Trans_mat = np.linalg.inv(Trans_mat)

    if points.shape[1] == 7:
        xyz = utils.points_to_homo(points[: , :3])
        
        points_w = Trans_mat @ xyz.T

        points_w = np.delete(points_w.T, -1, axis=1)
        heading = np.copy(points[:, -1])

        R = (Trans_mat)[:3, :3]
        
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular :
            z = math.atan2(R[1,0], R[0,0])
        else :
            z = 0
        return np.hstack((points_w, points[:, 3:-1], (heading + z).reshape(-1,1)))
    
    if points.shape[1] == 3:
        xyz = utils.points_to_homo(points)
        points_w = Trans_mat @ xyz.T
        points[: , :3] = np.delete(points_w.T, -1, axis=1)
        return points



def main():
    detection_path = "obj_boxes.pkl"
    data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
    sequence = "08"
    gt_poses = utils.get_gt_poses(data_path, sequence)

    detections = utils.load_pickle(detection_path)

    for idx, i in enumerate(detections.keys()):
        bbox = detections[i]['boxes']
        bbox = convert_bb_frame(bbox, gt_poses[idx], "global")    
        detections[i]['boxes'] = bbox
        detections[i]['labels'] = detections[i]['labels'].numpy().reshape(-1, 1)
    
    utils.save_pickle(detections, "global_obj_boxes.pkl")


main()