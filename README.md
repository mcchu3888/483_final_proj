# 483 Final Project: GATs for 2D-to-3D Pose Estimation in Baseball
Full visual results can be found here
A demo palyer that creates the 3D skeletal data from broadcast video is provided
Note that this repository does not contain any Hawk-Eye data as it is private 

Usage: to run the demo follow these steps:
1. Set up python virtual environment: python -m venv /path/to/new/virtual/environment and source venv/bin/activate
2. Install dependencies: pip3 install -r requirements.txt
3. Download model weights from here and upload to the demo folder
4. Run python3 demo.py inside the demo folder

The produced results are from the baseline, GAT, and STGAT models. There is also an isolated 2D keypoints video.
