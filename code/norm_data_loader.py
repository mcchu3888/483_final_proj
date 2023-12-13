import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

JOIN = os.path.join

pitch_lst = os.listdir("./data/3d_data")
path_2d = "./data/2d_data"
path_3d = "./data/3d_data"

def custom_min_max_scaler(data, min_vals, max_vals):
    data_range = max_vals - min_vals
    normalized_data = (data - min_vals) / data_range
    return normalized_data

min_values = np.array([-7, -12, -5])
max_values = np.array([7, 3, 4])

norm_min_values = np.inf 
norm_max_values = -np.inf

data_2d = []
data_3d = []
meta = []

for pitch in pitch_lst:
    pose_seq_2d = np.load(JOIN(path_2d, pitch))
    pose_seq_3d = np.load(JOIN(path_3d, pitch))
    pose_seq_3d -= pose_seq_3d[0, 8]
    
    outside_range = np.any((pose_seq_3d < min_values) | (pose_seq_3d > max_values))
    
    if outside_range:
        continue
    else:
        norm_pose_seq_3d = custom_min_max_scaler(pose_seq_3d, min_values, max_values)

    norm_min_values = np.minimum(norm_min_values, np.min(norm_pose_seq_3d, axis=(0, 1)))
    norm_max_values = np.maximum(norm_max_values, np.max(norm_pose_seq_3d, axis=(0, 1)))
    
    norm_pose_seq_3d = np.reshape(norm_pose_seq_3d, (norm_pose_seq_3d.shape[0], -1))

    meta.append(pitch)
    data_2d.append(pose_seq_2d)
    data_3d.append(norm_pose_seq_3d)

print("Normalized Pose Sequence 3D Shape:", norm_pose_seq_3d.shape)
print("Normalized Pose Sequence 3D Min Values:", norm_min_values)
print("Normalized Pose Sequence 3D Max Values:", norm_max_values)

meta = np.array(meta)
data_2d = np.array(data_2d)
data_3d = np.array(data_3d)

train_data, val_data, train_labels, val_labels, train_meta, val_meta = train_test_split(data_2d, data_3d, meta, test_size=0.2, random_state=42)

data_dir = "./data/norm_data/"

np.save(data_dir + "train_data.npy", train_data)
np.save(data_dir + "val_data.npy", val_data)
np.save(data_dir + "train_labels.npy", train_labels)
np.save(data_dir + "val_labels.npy", val_labels)
np.save(data_dir + "train_meta.npy", train_meta)
np.save(data_dir + "val_meta.npy", val_meta)
