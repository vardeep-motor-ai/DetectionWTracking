#!/usr/bin/env python3

from functools import partial
from functools import lru_cache
import glob
import time
from overlap import draw_3dbox, load_pickle
import click
import open3d as o3d
import sys

"""
Usage: python scripts/visualize_dataset.py "ply_files/*.ply" tracked_det
"""

# @lru_cache()
def o3d_read_geometry(filename, detection):
    o3d_boxes = draw_3dbox(detection, "red")

    pcd = o3d.io.read_point_cloud(filename)
    return  [pcd] + o3d_boxes


class Visualizer:
    def __init__(self, filenames, detection_keys, detections, sleep_time=5e-3):
        # Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        # self.render_options = self.vis.get_render_option().load_from_json("render_o3d11.json")
        self.render_options = self.vis.get_render_option()
        # Files to render
        self.files = filenames
        self.n_frames = len(filenames)

        self.detection_keys = detection_keys
        self.detections = detections
        # Initialize the default callbacks
        self._register_key_callbacks()

        # Add first frame
        self.idx = 0
        curr_det_path = self.detection_keys[self.idx]
        dets = self.detections[curr_det_path]['boxes']
        visual = o3d_read_geometry(self.files[self.idx], dets)

        for i in visual:
            self.vis.add_geometry(i)
        # self.vis.add_geometry(o3d_read_geometry(self.files[self.idx], dets))
        self.print_help()

        # Continous time plot
        self.stop = False
        self.sleep_time = sleep_time

    def next_frame(self, vis):
        self.idx = (self.idx + 1) % self.n_frames
        self.update_geometry()
        return False

    def prev_frame(self, vis):
        self.idx = (self.idx - 1) % self.n_frames
        self.update_geometry()
        return False

    def start_prev(self, vis):
        self.stop = False
        while not self.stop:
            self.next_frame(vis)
            time.sleep(self.sleep_time)

    def stop_prev(self, vis):
        self.stop = True

    def update_geometry(self):
        print("Visualizing {}".format(self.files[self.idx]), end="\r")
        curr_det_path = self.detection_keys[self.idx]
        dets = self.detections[curr_det_path]['boxes']
        self.vis.clear_geometries()
        visual = o3d_read_geometry(self.files[self.idx], dets)
        for i in visual:
            self.vis.add_geometry(i)
        
        # self.vis.add_geometry(
        #     o3d_read_geometry(self.files[self.idx]), reset_bounding_box=False
        # )
        self.vis.poll_events()
        self.vis.update_renderer()

    def set_render_options(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.render_options, key, value)

    def register_key_callback(self, key, callback):
        self.vis.register_key_callback(ord(str(key)), partial(callback))

    def set_white_background(self, vis):
        """Change backround between white and white."""
        self.render_options.background_color = [1.0, 1.0, 1.0]

    def set_black_background(self, vis):
        """Change backround between white and black."""
        self.render_options.background_color = [0.0, 0.0, 0.0]

    def _register_key_callbacks(self):
        self.register_key_callback("N", self.next_frame)
        self.register_key_callback("P", self.prev_frame)
        self.register_key_callback("S", self.start_prev)
        self.register_key_callback("X", self.stop_prev)
        self.register_key_callback("W", self.set_white_background)
        self.register_key_callback("B", self.set_black_background)

    def print_help(self):
        print("N: next")
        print("P: previous")
        print("S: start")
        print("X: stop")
        print("W: white background")
        print("B: black  background")
        print("Q: quit")

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


@click.command()
@click.argument("file_patterns", type=str)
@click.argument("name", type=str)
def visualizer(file_patterns, name):
    """Specify the filename pattern of the files you want to visualize."""
    detections_path = f"{name}.pkl"
    detections = load_pickle(detections_path)
    detection_keys = list(detections.keys())
    files = sorted(glob.glob(file_patterns))
    print("Visualizing {} models".format(len(files)))
    vis = Visualizer(files, detection_keys, detections)
    vis.run()


if __name__ == "__main__":
    visualizer()
