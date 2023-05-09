import os, numpy as np, sys
import pickle
from datetime import datetime


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def make_dataset(det_path, category, write_path, score_threshold=0.5):
	"""
	Get detections as pickle file from obj det and convert it in format:

	[frame, label, h, w, l, x, y, z, theta]
	"""
	
	det_str2id = {'Car':1, 'Pedestrian':2, 'Cyclist':3}
	cat = det_str2id[category]
	detections = load_pickle(det_path)
	dataset = np.array([])
	for frame, i in enumerate(detections.keys()):
		bb, labels, scores = detections[i].values()
		# bb, labels = detections[i].values()
		if (type(bb) ==  'torch.Tensor') and (labels.dim() != 2):
			bb, labels = bb.numpy(), labels.numpy().reshape(-1, 1)
		
		# Get objects of the class we want and filter them based on score

		# id_ = np.where((labels == cat) & (scores > score_threshold))[0]
		id_ = np.where((labels == cat))[0]
		# labels, scores, bb = labels[id_], scores[id_], bb[id_, :] 
		labels, bb = labels[id_], bb[id_, :] 
		bb_new = np.zeros_like(bb)
		bb_new[:, 3:] = bb[:, [0,1,2,-1]]
		bb_new[:, :3] = bb[:, [5,4,3]]
		
		n_frame = np.full_like(labels, frame).reshape(-1, 1)
		labels = labels.reshape(-1, 1)
		x = np.hstack((n_frame, labels, bb_new))
		dataset = np.append(dataset, x).reshape(-1, 9)
	
	save_file_path = os.path.join(write_path, category + ".txt")
	np.savetxt(save_file_path, dataset)
	return dataset

if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	if len(sys.argv) != 2:
		print("Please add the detection .pkl file path, it is in obj_boxes.pkl or global_obj_boxes.pkl something") 
		sys.exit(1)

	det_path = sys.argv[1]
	
	output_filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]
	time = datetime.now().strftime('%Y_%m_%d_%H_%M')

	det_id2str = {1:'Car', 2:'Pedestrian', 3:'Cyclist'}

	write_path = os.path.join("AB3DMOT/data", output_filename)
	if not os.path.isdir(write_path):
		os.makedirs(write_path)
		for cat in det_id2str.values():
			dataset = make_dataset(det_path, cat, write_path)