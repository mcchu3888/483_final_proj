# 483 Final Project: GATs for 2D-to-3D Pose Estimation in Baseball

This repository contains all the code I used to complete this project. It also contains a demo palyer that generates 3D skeletal data from broadcast video using the models described in the project. Note that this repository does not contain any Hawk-Eye data as this data is private to the New York Yankees and Major League Baseball.

Usage: to run the demo follow these steps:
1. Set up python virtual environment: python -m venv /path/to/new/virtual/environment and source venv/bin/activate
2. Install dependencies: pip3 install -r requirements.txt
3. Download model weights from here and upload to the demo folder
4. Run python3 demo.py inside the demo folder

The produced results are from the baseline, GAT, and STGAT models. There is also an isolated 2D keypoints video.
To run a video outside of the ones provided, please provide demo.py with the video url/path and the handedness of the pitcher (0 if righty, 1 if lefty).
