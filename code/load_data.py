import pandas as pd 
import numpy as np
import cv2 
import urllib.request
from ultralytics import YOLO
from michael.gnn.models.cv.rp_model import RPModel
from utils import *

hawkeye_df = pd.read_csv("./data/he_data.csv")
broadcast_df = pd.read_csv("./data/broadcast.csv").sample(frac=1)

pose_model = YOLO("./models/yolov8x-pose.pt")
rp_model = RPModel()

bad_pitches = 0
for index, row in broadcast_df.iterrows():
    url = row["url"]
    gameid = row["gameid"]
    eventseq = row["eventseq"]
    pitchseq = row["pitchseq"]
    islhp = row["islhp"]

    pitchid = f"{gameid}-{eventseq}-{pitchseq}"

    hawkeye_pitch = hawkeye_df[(hawkeye_df['gameid'] == gameid) & (hawkeye_df['eventseq'] == eventseq) & (hawkeye_df['pitchseq'] == pitchseq)]
    pose_seq_3d = []
    for i in range(1, 91):
        try:
            row = hawkeye_pitch[hawkeye_pitch["timeseq"] == i*10].iloc[0]
            elements = list(row['nose_x':'lear_z'])
            elements = np.reshape(elements, (int(len(elements) / 3), 3))
            elements[8] = [(x + y) / 2 for x, y in zip(elements[9], elements[12])]
            pose_seq_3d.append(elements)
        except:
            print(pitchid, i)
            break
    
    pose_seq_3d = np.array(pose_seq_3d)

    for i in range(1, pose_seq_3d.shape[0]):
        pose_seq_3d[i] = np.where(np.isnan(pose_seq_3d[i]), pose_seq_3d[i - 1], pose_seq_3d[i])
    
    if len(pose_seq_3d) != 90 or np.isnan(pose_seq_3d).any():
        bad_pitches += 1
        continue
    
    try:
        urllib.request.urlretrieve(url, "broadcast.mp4")
        results = pose_model.track("broadcast.mp4", save=False, tracker="bytetrack.yaml", imgsz=640, conf=0.25, persist=False)
        pitcherid = isolate_player_keypoints(results)["pitcher"]
        cap = cv2.VideoCapture("broadcast.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = round(fps/30)
        rp_idx = rp_model.find_rp(results, pitcherid, islhp, step)
        print(rp_idx, url)

    except Exception as e:
        bad_pitches += 1
        print(e)
        print(url)
        print("INFO: Can't find release point with confidence")
        continue

    pose_seq_2d = scale_fit(results, pitcherid, rp_idx, -60, 30, step)
    pose_seq_2d = np.nan_to_num(pose_seq_2d, nan=0)

    if len(pose_seq_2d) != 90 or len(pose_seq_3d) != 90:
        bad_pitches += 1
        print("lengths don't match")
        continue

    np.save(f"./data/2d_data/{pitchid}.npy", pose_seq_2d)
    np.save(f"./data/3d_data/{pitchid}.npy", pose_seq_3d)

print(bad_pitches)