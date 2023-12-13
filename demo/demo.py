import numpy as np
import torch
import cv2
import os
from ultralytics import YOLO
from utils import *
from models.rp_model import RPModel
from models.base import BaseModel
from models.gat import GAT_LSTM
from models.stgat import ST_GAT

# Here I have provided 3 demo videos :)
videos = [("https://sporty-clips.mlb.com/302d65a8-06d0-4c02-9b6b-2ce6cee4374c.mp4", 0)]
video_path, islhp = videos[0] # select pitch video to run
pitchid = "demo"

if not os.path.exists("./results"):
    os.mkdir("./results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {DEVICE}")
pose_model = YOLO("./weights/yolov8x-pose.pt")
pose_model.to(DEVICE)

results = pose_model.track(video_path, save=False, tracker="bytetrack.yaml", imgsz=640, conf=0.25, persist=True)
best_fits = isolate_player_keypoints(results)
pitcherid = best_fits["pitcher"]

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
step = round(fps/30)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"./results/{pitchid}_2d.mp4", fourcc, fps, (width, height))

for frame_count, result in enumerate(results):
    success, frame = cap.read()
    if not success:
        break

    for player, id in best_fits.items():
        for box, pose in zip(result.boxes, result.keypoints):
            if box.id == id:
                plot_skeleton_kpts(frame, pose.data[0], shape=(width, height), radius=3, line_thick=2)
                plot_one_box(box.xyxy[0], frame, (255, 0, 255), f'{player}', line_thickness=1)
    
    out.write(frame)
    percent_completed = ((frame_count + 1) / len(results)) * 100
    print(f"Processed {frame_count + 1} frames ({percent_completed:0.2f}%)")

rp_model = RPModel()
rp_idx = rp_model.find_rp(results, pitcherid, islhp, step)

print(f"INFO: Release Point Detected. Frame {rp_idx}")

pose_seq_2d = scale_fit(results, pitcherid, rp_idx, -60, 30, step)
pose_seq_2d = np.nan_to_num(pose_seq_2d, nan=0)
pose_seq_2d = np.expand_dims(pose_seq_2d, axis=0)
pose_seq_2d = torch.tensor(pose_seq_2d).double().to(DEVICE)

model = BaseModel().double().to(DEVICE)
model.load_state_dict(torch.load(f"./weights/base_model.pt"))
model.eval()
pred = model(pose_seq_2d)
pred = pred.detach().cpu().numpy()
reconstruction_3d("./results/demo_base.mp4", pred, "Base Model")


model = GAT_LSTM().double().to(DEVICE)
model.load_state_dict(torch.load(f"./weights/gat_model.pt"))
model.eval()
graph = model.make_graph(pose_seq_2d)
pred = model(graph.to(DEVICE))
pred = pred.detach().cpu().numpy()
reconstruction_3d("./results/demo_gat.mp4", pred, "GAT Model")


model = ST_GAT().double().to(DEVICE)
model.load_state_dict(torch.load(f"./weights/stgat_model.pt"))
model.eval()
graph = model.make_graph(pose_seq_2d)
pred = model(graph.to(DEVICE))
pred = pred.detach().cpu().numpy()
reconstruction_3d("./results/demo_stgat.mp4", pred, "STGAT Model")