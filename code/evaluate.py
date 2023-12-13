import torch
import numpy as np
import os
import sys
from models.base.base import BaseModel
from models.gat.gat import GAT_LSTM
from models.stgat.stgat import ST_GAT

def euclidean_dist(source, target):
    return np.sqrt(np.sum((source - target)**2))

min_values = np.array([-7, -12, -5])
max_values = np.array([7, 3, 4])

def inverse_min_max_scaling(scaled_values, min_vals, max_vals):
    original_values = scaled_values * (max_vals - min_vals) + min_vals
    original_values[:, :, :, 2] += 5
    original_values *= 12
    return original_values

joints = [
    "nose", "neck", 
    "rshoulder", "relbow", "rwrist", 
    "lshoulder", "lelbow", "lwrist", 
    "midhip", 
    "rhip", "rknee", "rankle", 
    "lhip", "lknee", "lankle", 
    "reye", "leye", 
    "rear", "lear"
]

JOIN = os.path.join
data_path = "./data/norm_data"
he_connections = [(0, 1), (2, 3), (3, 4), (2, 1), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), (16, 18)]
train_data = np.load(JOIN(data_path, "train_data.npy"), allow_pickle=True)
train_labels = np.load(JOIN(data_path, "train_labels.npy"), allow_pickle=True)
test_data = np.load(JOIN(data_path, "val_data.npy"), allow_pickle=True)
test_labels = np.load(JOIN(data_path, "val_labels.npy"), allow_pickle=True)

train_labels = train_labels.reshape(train_labels.shape[0], -1, 19, 3)
test_labels = test_labels.reshape(test_labels.shape[0], -1, 19, 3)
train_labels = inverse_min_max_scaling(train_labels, min_values, max_values)
test_labels = inverse_min_max_scaling(test_labels, min_values, max_values)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_results = []
gat_results = []
stgat_results = []

for i in range(3):
    base_model = BaseModel().double().to(device)
    base_model.load_state_dict(torch.load(f"./models/base/base_nc_{i}.pt"))
    base_model.eval()
    pred = base_model(torch.tensor(test_data).to(device))
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(pred.shape[0], -1, 19, 3)
    pred = inverse_min_max_scaling(pred, min_values, max_values)
    base_results.append(pred)

    gat_model = GAT_LSTM().double().to(device)
    gat_model.load_state_dict(torch.load(f"./models/gat/gat_nc_{i}.pt"))
    gat_model.eval()
    graph = gat_model.make_graph(torch.tensor(test_data).double())
    pred = gat_model(graph.to(device))
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(pred.shape[0], -1, 19, 3)
    pred = inverse_min_max_scaling(pred, min_values, max_values)
    gat_results.append(pred)

    stgat_model = ST_GAT().double().to(device)
    stgat_model.load_state_dict(torch.load(f"./models/stgat/stgat_nc_{i}.pt"))
    stgat_model.eval()
    graph = stgat_model.make_graph(torch.tensor(test_data).double())
    pred = stgat_model(graph.to(device))
    pred = pred.detach().cpu().numpy()
    pred = pred.reshape(pred.shape[0], -1, 19, 3)
    pred = inverse_min_max_scaling(pred, min_values, max_values)
    stgat_results.append(pred)

# test MSE and MPJPE
def calculate_mse(predictions, labels):
    mse_list = [np.mean((labels.flatten() - pred.flatten())**2) for pred in predictions]
    return np.mean(mse_list), np.std(mse_list)

# def calculate_mpjpe(predictions, labels):
#     label = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2], labels.shape[3])
#     runs_lst = []
#     for pred in predictions:
#         pred = pred.reshape(pred.shape[0]*pred.shape[1], pred.shape[2], pred.shape[3])
#         mpjpe_lst = []
#         for i in range(19):
#             joint_mean = np.mean([euclidean_dist(pt1, pt2) for pt1, pt2 in zip(pred[:, i::19].squeeze(), label[:, i::19].squeeze())])
#             mpjpe_lst.append(joint_mean)
#         runs_lst.append(np.mean(mpjpe_lst))

#     return np.mean(runs_lst), np.std(runs_lst)

# def calculate_mpjpe(predictions, labels):
#     label = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2], labels.shape[3])
#     runs_lst = []
#     for pred in predictions:
#         pred = pred.reshape(pred.shape[0]*pred.shape[1], pred.shape[2], pred.shape[3])
#         mpjpe_lst = []
#         for i in range(19):
#             joint_mean = np.mean([euclidean_dist(pt1, pt2) for pt1, pt2 in zip(pred[:, i::19].squeeze(), label[:, i::19].squeeze())])
#             mpjpe_lst.append(joint_mean)
#         runs_lst.append(mpjpe_lst)
#     return np.mean(runs_lst, axis=0), np.std(runs_lst, axis=0)

def calculate_mpjpe(predictions, labels):
    labels = labels[:, 60, :, :]
    runs_lst = []
    for pred in predictions:
        pred = pred[:, 60, :, :]
        mpjpe_lst = []
        for i in range(19):
            joint_mean = np.mean([euclidean_dist(pt1, pt2) for pt1, pt2 in zip(pred[:, i::19].squeeze(), labels[:, i::19].squeeze())])
            mpjpe_lst.append(joint_mean)
        runs_lst.append(mpjpe_lst)
    return np.mean(runs_lst, axis=0), np.std(runs_lst, axis=0)

base_mse_mean, base_mse_std = calculate_mse(base_results, test_labels)
gat_mse_mean, gat_mse_std = calculate_mse(gat_results, test_labels)
stgat_mse_mean, stgat_mse_std = calculate_mse(stgat_results, test_labels)

base_mpjpe_mean, base_mpjpe_std = calculate_mpjpe(base_results, test_labels)
gat_mpjpe_mean, gat_mpjpe_std = calculate_mpjpe(gat_results, test_labels)
stgat_mpjpe_mean, stgat_mpjpe_std = calculate_mpjpe(stgat_results, test_labels)


# print("Base Model:")
# print(f"MSE: {base_mse_mean:.1f}$\pm${base_mse_std:.1f}")
# print(f"MPJPE: {base_mpjpe_mean:.1f}$\pm${base_mpjpe_std:.1f}")

# print("\nGAT Model:")
# print(f"MSE: {gat_mse_mean:.1f}$\pm${gat_mse_std:.1f}")
# print(f"MPJPE: {gat_mpjpe_mean:.1f}$\pm${gat_mpjpe_std:.1f}")

# print("\nSTGAT Model:")
# print(f"MSE: {stgat_mse_mean:.1f}$\pm${stgat_mse_std:.1f}")
# print(f"MPJPE: {stgat_mpjpe_mean:.1f}$\pm${stgat_mpjpe_std:.1f}")

print("Base Model:")
for mean, std, joint in zip(base_mpjpe_mean, base_mpjpe_std, joints):
    print(f"{mean:.1f}$\pm${std:.1f}", end =" & ")

print("\nGAT Model:")
for mean, std, joint in zip(gat_mpjpe_mean, gat_mpjpe_std, joints):
    print(f"{mean:.1f}$\pm${std:.1f}", end =" & ")

print("\nSTGAT Model:")
for mean, std, joint in zip(stgat_mpjpe_mean, stgat_mpjpe_std, joints):
    print(f"{mean:.1f}$\pm${std:.1f}", end =" & ")