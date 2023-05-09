import os
import open3d as o3d
import numpy as np

from overlap import custom_visualization, draw_3dbox, load_pickle

def main():
    pc_path = "/home/sandhu/project/p04-masterproject/global_ply"
    pc_file_path_list = sorted([os.path.join(pc_path, i) for i in os.listdir(pc_path)])
    obj_path = "/home/sandhu/project/p04-masterproject/pkl_files/global_objects_after_posechange.pkl"
    objects = load_pickle(obj_path)
    obj_keys = list(objects.keys())
    
    for idx, _ in enumerate(pc_file_path_list):
        pc_1 = o3d.io.read_point_cloud(pc_file_path_list[idx])
        pc_2 = o3d.io.read_point_cloud(pc_file_path_list[idx+1])
        pc_3 = o3d.io.read_point_cloud(pc_file_path_list[idx+2])
        # not_car_1 = o3d.io.read_point_cloud("/home/sandhu/project/p04-masterproject/000035.pcd")
        # not_car_2 = o3d.io.read_point_cloud("/home/sandhu/project/p04-masterproject/000036.pcd")
        # not_car_3 = o3d.io.read_point_cloud("/home/sandhu/project/p04-masterproject/000037.pcd")
        # merged_car = o3d.io.read_point_cloud("/home/sandhu/project/p04-masterproject/car_2.pcd")
        # idx = 36
        print(pc_file_path_list[idx])
        obj_1 = objects[obj_keys[idx]]['boxes']
        obj_2 = objects[obj_keys[idx+1]]['boxes']
        obj_3 = objects[obj_keys[idx+2]]['boxes']
        
        box_1 = draw_3dbox(obj_1, "red")
        box_2 = draw_3dbox(obj_2, "green")
        box_3 = draw_3dbox(obj_3, "yellow")

        # merged_car.paint_uniform_color([1, 0, 0])
        
        # not_car_1.paint_uniform_color([1, 1, 1])
        # not_car_2.paint_uniform_color([1, 1, 1])
        # not_car_3.paint_uniform_color([1, 1, 1])
        pc_1.paint_uniform_color([1, 1, 1])
        pc_2.paint_uniform_color([1, 1, 1])
        pc_3.paint_uniform_color([1, 1, 1])
        
        # custom_visualization([not_car_1,not_car_2,not_car_3, merged_car] + box_1 + box_2 + box_3, "hello")
        custom_visualization([pc_1, pc_3, pc_2]+ box_1, "Hello")

main()