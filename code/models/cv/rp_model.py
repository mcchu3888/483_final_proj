import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import *
import os
from sklearn.preprocessing import MinMaxScaler
import sys

JOIN = os.path.join

# Takes a YOLO pose estimation results object and returns the index of the frame of release
class RPModel:
    def __init__(self):
        self.pad_val = 0
        self.seq_len = 450
        with tf.device('/cpu:0'):
            self.rp_model = load_model("/home/ubuntu/michael/yolo_pe/models/rp_lstm.keras")
    
    # finds the release point frame
    def find_rp(self, results, id, islhp, step, threshold=0.01):
        """Input: results: YOLO results object from tracking
                    id: the YOLO id of the desired person/player
                    islhp: 0 for right handed pitchers, 1 for lefties
                    step: stride required to get to 30 fps, ex 60 fps -> step = 2
                    threshold: minimum sigmond probability acceptable to be release point"""

        pose_seq = []
        for i in range(self.seq_len):
            fit_exists = False
            frame_idx = i*step

            if not frame_idx >= len(results):
                result = results[i*step]
                for box, pose in zip(result.boxes, result.keypoints):
                    if box.id == id:
                        xy_keypoints = pose.xy[0].cpu().numpy()
                        xy_keypoints[xy_keypoints==0] = np.nan
                        xy_keypoints = MinMaxScaler().fit_transform(xy_keypoints)
                        pose_seq.append(xy_keypoints)
                        fit_exists = True
                        break

            if not fit_exists:
                pose_seq.append(np.nan * np.ones((17, 2)))
        
        pose_seq = np.array(pose_seq)
        seq_len, keypoints, features = pose_seq.shape
        pose_seq = np.reshape(pose_seq, (seq_len, keypoints*features))
        pose_seq = np.concatenate((pose_seq, np.expand_dims(islhp*np.ones(self.seq_len), axis=1)), axis=1)
        pose_seq = np.nan_to_num(pose_seq, nan=self.pad_val)
        pose_seq = np.expand_dims(pose_seq, axis=0)

        output = self.rp_model.predict(pose_seq).squeeze()

        max_idx = np.argmax(output)

        if output[max_idx] > threshold:
            return max_idx
        else:
            raise Exception("Cannot find release with confidence")