import sys
import numpy as np
import pickle 
import os

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(pkl_file, path):
    with open(path, 'wb') as f:
        pickle.dump(pkl_file, f)

def make_pkl(old_pkl_path, dataset):
    pkl_file = load_pickle(old_pkl_path)
    new_pkl = {}
     
    for idx, frame in enumerate(pkl_file):
        data_per_frame = dataset[dataset[:, 0] == idx]
        if len(data_per_frame) == 0:
            # If there are no detections after filtering then give empty list
            new_pkl[frame] = {"boxes": [], "labels" : []}

        labels = data_per_frame[:, 1]
        bb = data_per_frame[:, 2:]
        # bb = np.zeros((len(data_per_frame), 7))
        # bb[:, [0,1,2,-1]] = data_per_frame[:, 5:]
        # bb[:, [5,4,3]] = data_per_frame[:, 2:5]
        new_pkl[frame] = {"boxes": bb, "labels" : labels}

    return new_pkl


def make_single_dataset(tracked_traj_path):
    traj_path_list = [os.path.join(tracked_traj_path, file) for file in os.listdir(tracked_traj_path)]
    dataset = np.array([])
    for i in traj_path_list:
        traj = np.loadtxt(i)
        traj = traj[:, [0,2,10,11,12,13,14,15,16]]
        dataset = np.append(dataset, traj).reshape(-1, 9)
        
    dataset = dataset[dataset[:, 0].argsort()]
    return dataset

def main():
    
    if len(sys.argv) != 2:
        print("Please add the path of tracking text results like [AB3DMOT/results/2022_01_26_13_28]") 
        sys.exit(1)

    np.set_printoptions(suppress=True)
    save_path_pkl = sys.argv[1] +".pkl"
    print(save_path_pkl)
    old_pkl_path = "pkl_files/global_objects_after_posechange.pkl"
    
    tracked_traj_path = sys.argv[1]

    # Categories
    det_id2str = {1:'Car', 2:'Pedestrian', 3:'Cyclist'}

    # This dataset has info like [frame, category, and obj det stuff]
    dataset = make_single_dataset(tracked_traj_path)
    new_pkl = make_pkl(old_pkl_path, dataset)
    
    save_pkl(new_pkl, save_path_pkl)

if __name__ == "__main__":
    main()