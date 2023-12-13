import numpy as np
import pandas as pd
import torch
import cv2
import os
from ultralytics import YOLO
from utils import *

video_path = "https://sporty-clips.mlb.com/302d65a8-06d0-4c02-9b6b-2ce6cee4374c.mp4"
pitchid = "1542104-2860-1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {DEVICE}")
pose_model = YOLO("./models/cv/yolov8x-pose.pt")
pose_model.to(DEVICE)

results = pose_model.track(video_path, save=False, tracker="bytetrack.yaml", imgsz=640, conf=0.25, persist=True)
best_fits = isolate_player_keypoints(results)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"./output/{pitchid}.mp4", fourcc, fps, (width, height))

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

