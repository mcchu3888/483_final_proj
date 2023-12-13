import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from models.base.base import BaseModel
from models.gat.gat import GAT_LSTM
from models.stgat.stgat import ST_GAT

JOIN = os.path.join
data_path = "./data/norm_data"
he_connections = [(0, 1), (2, 3), (3, 4), (2, 1), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), (16, 18)]
train_data = np.load(JOIN(data_path, "train_data.npy"), allow_pickle=True)
train_labels = np.load(JOIN(data_path, "train_labels.npy"), allow_pickle=True)
test_data = np.load(JOIN(data_path, "val_data.npy"), allow_pickle=True)
test_labels = np.load(JOIN(data_path, "val_labels.npy"), allow_pickle=True)
train_meta = np.load(JOIN(data_path, "train_meta.npy"), allow_pickle=True)
test_meta = np.load(JOIN(data_path, "val_meta.npy"), allow_pickle=True)

model_name = "stgat"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ST_GAT().double().to(device)
model.load_state_dict(torch.load(f"./models/{model_name}/{model_name}_model.pt"))
model.eval()
graph = model.make_graph(torch.tensor(test_data).double())
pred = model(graph.to(device))
# pred = model(torch.tensor(test_data).to(device))
pred = pred.detach().cpu().numpy()

min_values = np.array([-7, -12, -5])
max_values = np.array([7, 3, 4])

def inverse_min_max_scaling(scaled_values, min_vals, max_vals):
    original_values = scaled_values * (max_vals - min_vals) + min_vals
    original_values[:, :, :, 2] += 5
    return original_values

test_labels = test_labels.reshape(test_labels.shape[0], -1, 19, 3)
pred = pred.reshape(pred.shape[0], -1, 19, 3)

pred = inverse_min_max_scaling(pred, min_values, max_values)
test_labels = inverse_min_max_scaling(test_labels, min_values, max_values)

output_path = f"/home/ubuntu/michael/gnn/output/{model_name}_overlays/"
data_count = len(test_data)
width = 640
height = 480 
fps = 30

for pitch_idx, (pred_pitch, label, meta) in enumerate(zip(pred, test_labels, test_meta)):
    meta = meta[:-4]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path + f"{meta}_{model_name}.mp4", fourcc, fps, (width, height))

    for frame_idx, (frame, actual) in enumerate(zip(pred_pitch, label)):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(-7, 7)
        ax.set_zlim(0, 9)
        ax.set_ylim(-12, 3)

        elements = actual
        ax.scatter3D(elements[:, 0], elements[:, 1], elements[:, 2], color='red')
        for idx1, idx2 in he_connections:
            ax.plot3D([elements[:, 0][idx1], elements[:, 0][idx2]], [elements[:, 1][idx1], elements[:, 1][idx2]], [elements[:, 2][idx1], elements[:, 2][idx2]], color='red')
        
        elements = frame
        ax.scatter3D(elements[:, 0], elements[:, 1], elements[:, 2], color="blue")
        for idx1, idx2 in he_connections:
            ax.plot3D([elements[:, 0][idx1], elements[:, 0][idx2]], [elements[:, 1][idx1], elements[:, 1][idx2]], [elements[:, 2][idx1], elements[:, 2][idx2]], color='blue')
        
        ax.plot([], [], '-', color='red', label='Hawk-Eye')
        ax.plot([], [], '-', color='blue', label='Predicted')
        ax.legend(loc="upper left")

        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape((height, width, 3))

        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        out.write(image_array_bgr)

        plt.close(fig)
        print(f"Processed frame {frame_idx+1} of 90")

    progress = (pitch_idx + 1) / data_count * 100
    print(f"INFO: Processed {pitch_idx + 1} pitch ({progress:.2f}%).")

    out.release()