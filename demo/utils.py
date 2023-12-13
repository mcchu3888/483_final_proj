import numpy as np
import torch
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

CLASSES = {'batter': 0, 'catcher': 1, 'other': 2, 'pitcher': 3, 'umpire': 4}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def isolate_player_keypoints(results, stride = 20):
    pos_cls = YOLO("/home/ubuntu/michael/yolo_pe/models/yolov8-pos.pt")
    cam_cls = YOLO("/home/ubuntu/michael/yolo_pe/models/yolov8-cam.pt")
    pos_cls.to(DEVICE)
    cam_cls.to(DEVICE)

    players = {"batter": [], "catcher": [], "pitcher": []}

    for frame_idx, result in enumerate(results):
        if frame_idx % stride == 0:
            frame_type_cls =  cam_cls.predict(result.orig_img, imgsz=640, verbose=False)[0].probs.top1 
            if frame_type_cls == 1:
                players_frame = {"batter": [], "catcher": [], "pitcher": []}
                for box in result.boxes:
                    minx, miny, maxx, maxy = [int(x) if x >= 0 else 0 for x in box.xyxy[0].tolist()]
                    cropped_person = result.orig_img[miny:maxy, minx:maxx, ...]
                    try:
                        cls_pred = pos_cls.predict(cropped_person, imgsz=480, verbose=False)
                        for key in players_frame.keys():
                            prob = cls_pred[0].probs.data[CLASSES[key]]
                            players_frame[key].append(prob)
                    except:
                        print(minx, miny, maxx, maxy, box.xyxy[0])
                        cv2.imwrite("test.jpg", cropped_person)

                for key in players_frame.keys():
                    if players_frame[key]:
                        probs = players_frame[key]
                        best_fit = probs.index(max(probs))
                        players[key].append(result.boxes[best_fit].id)
    
    for key, value in players.items():
        counter = Counter(players[key])
        try:
            id, count = counter.most_common()[0]
        except:
            players[key] = -1
        else:
            players[key] = id
    
    return players


def plot_skeleton_kpts(im, kpts, radius=5, shape=(640, 640), confi=0.25, line_thick=2):
    pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                            [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                            [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                            dtype=np.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    _, ndim = kpts.shape
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]]
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < confi:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    ndim = kpts.shape[-1]
    for i, sk in enumerate(skeleton):
        pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
        pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
        if ndim == 3:
            conf1 = kpts[(sk[0] - 1), 2]
            conf2 = kpts[(sk[1] - 1), 2]
            if conf1 < confi or conf2 < confi:
                continue
        if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
            continue
        cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=line_thick, lineType=cv2.LINE_AA)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def scale_fit(results, id, ref_idx, start, stop, step):
    ref_idx *= step
    pose_seq = []
    for idx in range(ref_idx + start*step, ref_idx + stop*step, step):
        fit_exists = False
        if not idx < 0 and not idx >= len(results):
            result = results[idx]
            for box, pose in zip(result.boxes, result.keypoints):
                if box.id == id:
                    xy_keypoints = pose.xy[0].cpu().numpy()
                    xy_keypoints[xy_keypoints==0] = np.nan
                    xy_keypoints = MinMaxScaler().fit_transform(xy_keypoints)
                    pose_seq.append(xy_keypoints)
                    fit_exists = True
        
        if not fit_exists:
            pose_seq.append(np.nan * np.ones((17, 2)))
    
    pose_seq = np.array(pose_seq)
    seq_len, keypoints, features = pose_seq.shape
    pose_seq = np.reshape(pose_seq, (seq_len, keypoints*features))
    return pose_seq

def reconstruction_3d(file, pred, title):
    min_vals = np.array([-7, -12, -5])
    max_vals = np.array([7, 3, 4])
    he_connections = [(0, 1), (2, 3), (3, 4), (2, 1), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), (16, 18)]

    pred = pred.reshape(pred.shape[0], -1, 19, 3)
    pred = pred * (max_vals - min_vals) + min_vals
    pred[:, :, :, 2] += 5

    width = 640
    height = 480 
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file, fourcc, fps, (width, height))

    for frame_idx, frame in enumerate(pred[0]):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(-7, 7)
        ax.set_zlim(0, 9)
        ax.set_ylim(-12, 3)

        elements = frame
        ax.scatter3D(elements[:, 0], elements[:, 1], elements[:, 2], color="blue")
        for idx1, idx2 in he_connections:
            ax.plot3D([elements[:, 0][idx1], elements[:, 0][idx2]], [elements[:, 1][idx1], elements[:, 1][idx2]], [elements[:, 2][idx1], elements[:, 2][idx2]], color='red')
        
        plt.title(title)
        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape((height, width, 3))

        image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        out.write(image_array_bgr)

        plt.close(fig)
        print(f"Processed frame {frame_idx+1} of 90")

    print(f"INFO: Finish creating results for {title}")
    out.release()