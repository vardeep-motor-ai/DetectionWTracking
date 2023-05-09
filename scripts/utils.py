import math
import numpy as np
import pickle 
from OpenPCDet.tools.visual_utils import visualize_utils
import os
import pykitti 
import open3d as o3d

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def points2pcd(points, intensities):
    pcd = o3d.geometry.PointCloud()
    colors = np.full_like(points, intensities.reshape(-1,1))

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def getting_point_mask(boxes, points):
    # print("Starting Masking of points inside the overlap box")
    segmented_idx = np.zeros(len(points), dtype=bool)
    pcd_masks = []
    for box in boxes:
        mask = points_in_box(box.reshape(1, -1), points)
        segmented_idx = segmented_idx + mask
        indices = mask.nonzero()[0]
        points_inside = points[indices]
        pcd_mask = points2pcd(points_inside, np.ones(len(points_inside)))
        pcd_mask.paint_uniform_color(np.array([1, 1, 0]))
        pcd_masks.append(pcd_mask)
    segmented_idx = segmented_idx + 1
    for idx, i in enumerate(segmented_idx):
        if i == 1:
            pass
        elif i == 2:
            segmented_idx[idx] = 251
    return pcd_masks, segmented_idx


def points_in_box(box, points: np.ndarray):
    """
    Checks whether points are inside the box.
    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """
    corners = visualize_utils.boxes_to_corners_3d(box).squeeze()

    p1 = corners[2]
    p_x = corners[3]
    p_y = corners[1]
    p_z = corners[6]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1
    
    iv = np.dot(v, i)
    jv = np.dot(v, j)
    kv = np.dot(v, k)


    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    return mask


def bin2array(file):
    array = np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    points = array[:, :3]
    intensity = array[:, -1]
    return points, intensity


def saving_predictions(segmented_idx, lidar_path, save_dir):
    fullname = os.path.basename(lidar_path)
    name, _ = os.path.splitext(fullname)
    sem_label = segmented_idx.astype(np.uint16)
    inst_label = np.zeros_like(sem_label, dtype=np.uint32)
    label = sem_label + inst_label
    save_path = f"results/{save_dir}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    label.tofile(f"{save_path}/{name}.label")
    return 

def bb_order_xyz(traj):
    traj = traj[:, [10,11,12,13,14,15,16]]
    bb = np.zeros((len(traj), 7))
    bb[:, [0,1,2,-1]] = traj[:, 3:]
    bb[:, [5,4,3]] = traj[:, :3]
    return bb

def get_gt_poses(data_path: str, sequence: str):
    
    """
    This function loads the GT poses on the Velodyne coordinate frame.
    """
    
    data = pykitti.odometry(data_path, sequence)
    poses = data.poses
    T_cam0_w0 = np.linalg.inv(poses[0])

    T_cam_velo = data.calib.T_cam0_velo
    T_velo_cam = np.linalg.inv(T_cam_velo)
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(T_cam0_w0).dot(pose).dot(T_cam_velo))
    # gt_poses = T_velo_cam @ data.poses @ T_cam_velo
    gt_poses = np.stack(new_poses)
    return gt_poses


def points_to_homo(points):
    ones = np.ones(len(points)).reshape(-1,1)
    points = np.hstack((points, ones))
    return points

def draw_3dbox(bbox_coords, color):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    This is how the corners are retured 
    Input: NX7 Bounding box corrds
    Output: Open3d Lineset object
    """
    visuals = []

    points = visualize_utils.boxes_to_corners_3d(bbox_coords)
    lines = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    
    if color == "red":
        colors = [[1, 0, 0] for i in range(len(lines))]
    if color == "something":
        colors = [[0, 1, 0] for i in range(len(lines))]
    for i in points:
        line_set = o3d.geometry.LineSet()
        line_set.points =  o3d.utility.Vector3dVector(i)
        line_set.lines =  o3d.utility.Vector2iVector(lines)
        line_set.colors =  o3d.utility.Vector3dVector(colors)
        visuals.append(line_set)
    return visuals


def points2pcd(points, intensities):
    pcd = o3d.geometry.PointCloud()
    colors = np.full_like(points, intensities.reshape(-1,1))

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def custom_visualization(visual, window_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for i in visual:
        vis.add_geometry(i)
    vis.get_render_option().load_from_json("scripts/render_o3d.json")
    vis.run()
    vis.destroy_window()

def save_pickle(pickle_file, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        xyz = points_to_homo(points[: , :3])
        
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
        xyz = points_to_homo(points)
        points_w = Trans_mat @ xyz.T
        points[: , :3] = np.delete(points_w.T, -1, axis=1)
        return points