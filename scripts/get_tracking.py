import os
import numpy as np
import sys
from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
import argparse


def save_tracking_file(det_files, save_dir):
    # dataset is the txt data file of detections
    mot_tracker = AB3DMOT()
    np.set_printoptions(suppress=True)
    # eval_file = os.path.join(save_dir + '_eval.txt')
    # eval_file = open(eval_file, 'w')

    for idx, frame in enumerate(det_files):
        tracked_dets = []
        # dets are the detections in that frame of format [h, w, l, x, y, z, theta]
        dets = np.load(frame)
        dets[:, [5, 4, 3, 0, 1, 2, 6]] = dets[:, [0, 1, 2, 3, 4, 5, 6]]
    # This additional infor contains [label of object, [2d bb], score, and alpha]
        additional_info = np.zeros_like(dets)
        dets_all = {'dets': dets, 'info': additional_info}
        trackers = mot_tracker.update(dets_all)

        for d in trackers:

            # h, w, l, x, y, z, theta in camera coordinate
            bbox3d_tmp = d[0:7]
            id_tmp = d[7]
            ori_tmp = d[8]
            type_tmp = 0
            bbox2d_tmp_trk = d[10:14]
            conf_tmp = d[14]

    # Saved tracked results below are in format:
        # [frame, id, obj_class, 7X[zeros], x, y, z, l, w, h, theta]
            one_track = [bbox3d_tmp[5], bbox3d_tmp[4], bbox3d_tmp[3],
                         bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2],  bbox3d_tmp[6]]
            tracked_dets.extend(one_track)

            # str_to_srite = '%d %d %d 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (idx, id_tmp,
            #                                                                           type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[
            #                                                                               1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3],
            #                                                                           bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[
            #                                                                               2], bbox3d_tmp[1], bbox3d_tmp[0],  bbox3d_tmp[6],
            #                                                                           conf_tmp)
            # eval_file.write(str_to_srite)
        tracked_det = np.array(tracked_dets).reshape(-1, 7)
        np.save(f"{save_dir}/{(str(idx).zfill(4))}", tracked_det)
        # if frame == min_frame+5:
        # break
    # eval_file.close()


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--save_path', '-s', type=str, default=None, required=True,
                        help='specify the path where you want to save the output')
    parser.add_argument('--det_path', '-d', type=str, default=None, required=True,
                        help='specify the detection directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_config()
    detection_path = args.det_path
    assert os.path.isdir(detection_path), "This dir does not exist"

    save_root = os.path.basename(args.save_path)

    if not os.path.isdir(save_root):
        os.makedirs(save_root)

    detection_files = sorted([os.path.join(detection_path, file)
                              for file in os.listdir(detection_path)])
    save_tracking_file(detection_files, save_root)
