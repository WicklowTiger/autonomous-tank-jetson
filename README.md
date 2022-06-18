# autonomous-tank-jetson
Jetson Nano part of Autonomous Targeting System project.

### Usage (Python 3.6.9 required!!!)
- Install YOLOv5 version 6.0 with its requirements from https://github.com/ultralytics/yolov5/tree/v6.0
- Replace datasets.py and detect.py with the provided datasets.py and run_detect.py
- Add best.pt to yolo folder
- Make a script which runs:
  - export OPENBLAS_CORETYPE=ARMV8
  - sudo -E python3 --weights best.pt --source 0 --img 640 --view-img
- chmod +x script_name
- ./script_name to run live detection
