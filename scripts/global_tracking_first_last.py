import sys
import numpy as np
import os
import math
import utils
from tqdm import tqdm

class TrackedDataset():
    def __init__(self, path, distance_thresh, min_traj_len):
        self.min_traj_len = min_traj_len
        self.distance_thresh = distance_thresh
        self.path = path
        self.tracked_object = np.concatenate([np.loadtxt(i) for i in self.path])
        self.filter_static_moving_id()
        print(len(self.moving_id))
        self.frames = np.unique(self.tracked_object[:, 0]) 


    def filter_static_moving_id(self):
        """
        The loaded dataset should be of format [frame, id, obj_class, 7X[zeros], x, y, z, l, w, h, theta]
        """
        max_id, min_id = np.max(self.tracked_object[:, 1]), np.min(self.tracked_object[:, 1])
        self.moving_id = []
        self.static_id = []
        self.empty_id = []
        total_ids = max_id 

        for id in range(int(min_id), int(max_id+1)):
            trk = self.tracked_object[self.tracked_object[:, 1] == id]
            xy = trk[:, 10:12]
            if len(xy) <= self.min_traj_len:
                # These are the empty trajs 
                self.empty_id.append(id)
                continue
            distance = math.sqrt(((xy[0][0]-xy[-1][0])**2)+((xy[0][1]-xy[-1][1])**2))
                
            if distance > self.distance_thresh:
                print("*" * 20)
                print(id)
                print(distance)
                print(xy[0])
                print(xy[-1])
                print(len(xy))
                # if id in np.arange(727, 735, step=1): 
                    # print(id , distance)
                    # print(xy)
                self.moving_id.append(id)
            else:
                self.static_id.append(id)
        print("Empty ID length", len(self.empty_id))
        print("Static ID length",len(self.static_id))
        print("Moving ID length",len(self.moving_id))
        assert (len(self.moving_id)+len(self.empty_id)+len(self.static_id)) == max_id, "Oops"

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
    """
    This file expects as input globally tracked trajs i.e. traj which are tracked in global frame
    It will get those tracks and detect moving and non moving objects based on first and last frame
    Usage: python scripts/global_tracking_first_last.py [AB3DMOT/results/ something] [yes]
    where the path repersents where the resulting globally tracked traj are 
    2nd part [yes] means if we want to convert the lidar points to global frame or not, which we do 
    """
    np.set_printoptions(suppress=True)

    if len(sys.argv) != 4:
        print("Usage: python first_n_last.py [.txt tracking_file_path] [yes/no to transform boxes or not] [object detections path]")
        sys.exit(1)
    
    output_dir = os.path.basename(sys.argv[1])
    
    if sys.argv[2] == "yes":
        data_path = "/media/sandhu/SSD/dataset/kitti/odometry/dataset"
        sequence = output_dir
        gt_poses = utils.get_gt_poses(data_path, sequence)
    
    output_dir = os.path.basename(sys.argv[1])
    tracked_dataset_path = sys.argv[1]    
    tracked_files_path = [os.path.join(tracked_dataset_path, file) for file in os.listdir(tracked_dataset_path)]
    
    lidar_files = list(utils.load_pickle(sys.argv[3]).keys())
    # Here adding list of paths of .txt files. This should be chamged in future
    
    car = TrackedDataset(tracked_files_path, distance_thresh=5, min_traj_len=10)

    pbar = tqdm(enumerate(range(len(lidar_files))), total=len(lidar_files))
    
    for id, frame in pbar:
        # id = 100
        # frame = 100
        car_moving_data_inframe = car.frame_moving_data(frame)
        
        # print(car_moving_data_inframe)
        # break
        all_moving_data = car_moving_data_inframe
        
        boxes = all_moving_data[:, -8:-1] 
        points, _ = utils.bin2array(lidar_files[id])
        
        
        # convert points to global frame as well as boxes are already in global frame        
        if sys.argv[2] == "yes":
            points = convert_bb_frame(points, gt_poses[id], "global")    

        # pcd = utils.points2pcd(points, np.full((1, len(points)), 0.5))
        # o3d_box = utils.draw_3dbox(boxes, "red")
        # to_draw = [pcd] + o3d_box
        # utils.custom_visualization(to_draw, "Name")

        pcd_mask, segmented_idx = utils.getting_point_mask(boxes, points)
        
        # sanity check
        assert len(points) == len(segmented_idx), "Points and labels size does not match"
        
        utils.saving_predictions(segmented_idx, lidar_files[id], output_dir)
        # utils.custom_visualization(to_draw + pcd_mask, "Name")
    

if __name__ == "__main__":
    main()