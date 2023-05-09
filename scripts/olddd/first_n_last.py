import sys
import numpy as np
import os
import math
import utils
from tqdm import tqdm

class TrackedDataset():
    def __init__(self, path):
        self.path = path
        self.tracked_object = np.loadtxt(self.path)
        self.filter_static_moving_id()
        print(len(self.moving_id))
        self.frames = np.unique(self.tracked_object[:, 0]) 

    def filter_static_moving_id(self):
        """
        The loaded dataset should be of format [frame, id, obj_class, 7X[zeros], x, y, z, l, w, h, theta]
        """
        self.tracked_object = np.loadtxt(self.path)
        max_id, min_id = np.max(self.tracked_object[:, 1]), np.min(self.tracked_object[:, 1])
        self.moving_id = []
        self.static_id = []
        for id in range(int(min_id), int(max_id+1)):
            trk = self.tracked_object[self.tracked_object[:, 1] == id]
            
            xy = trk[:, 10:12]
            if len(xy) < 2:
                continue
            distance = math.sqrt(((xy[0][0]-xy[-1][0])**2)+((xy[0][1]-xy[-1][1])**2))
            if distance > 3:
                self.moving_id.append(id)
            else:
                self.static_id.append(id) 

    def frame_moving_data(self, frame):
        frame_data = self.tracked_object[self.tracked_object[:, 0] == frame]
        frame_ids = frame_data[:, 1]
        moving_id_inframe = [id for id in frame_ids if id in self.moving_id]
        moving_data = []
        for id in moving_id_inframe:
            idx = np.where(frame_data[:, 1] == id)[0]
            moving_data.append(frame_data[idx])
        if len(moving_data) == 0:
            return np.array([]).reshape(0,18)
        return np.concatenate(moving_data)

def convert_bb_to_current_frame(points, T_0_current):
    
    if points.shape[1] == 7:
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
    
    if points.shape[1] == 3:
        xyz = utils.points_to_homo(points)
        points_w = T_0_current @ xyz.T
        points[: , :3] = np.delete(points_w.T, -1, axis=1)
        return points



def main():
    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3:
        print("Usage: python first_n_last.py [tracking_file_path] [yes/no to transform boxes or not]")
        sys.exit(1)
    
    if sys.argv[2] == "yes":
        data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
        sequence = "08"
        gt_poses = utils.get_gt_poses(data_path, sequence)

    tracked_dataset_path = sys.argv[1]    
    tracked_files_path = [os.path.join(tracked_dataset_path, file) for file in os.listdir(tracked_dataset_path)]
    
    lidar_files = list(utils.load_pickle("obj_boxes.pkl").keys())
    
    car = TrackedDataset(tracked_files_path[0])

    pbar = tqdm(enumerate(range(len(lidar_files))), total=len(lidar_files))
    
    for id, frame in pbar:
        
        car_moving_data_inframe = car.frame_moving_data(frame)

        all_moving_data = car_moving_data_inframe
        
        boxes = all_moving_data[:, -8:-1] 
        points, _ = utils.bin2array(lidar_files[id])
        # convert points to global frame as well as boxes are already in global frame        
        if sys.argv[2] == "yes":
            points = convert_bb_to_current_frame(points, gt_poses[id])

        pcd = utils.points2pcd(points, np.full((1, len(points)), 0.5))
        o3d_box = utils.draw_3dbox(boxes, "red")
        to_draw = [pcd] + o3d_box
        # utils.custom_visualization(to_draw, "Name")

        pcd_mask, segmented_idx = utils.getting_point_mask(boxes, points)
        
        # sanity check
        assert len(points) == len(segmented_idx), "Points and labels size does not match"
        
        utils.saving_predictions(segmented_idx, lidar_files[id], "tracked_first_last")
        # utils.custom_visualization(to_draw + pcd_mask, "Name")
        
        # if id == 10:
            # break

if __name__ == "__main__":
    main()