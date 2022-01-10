import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
resW, resH = '1280', '720'
imW, imH = int(resW), int(resH)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to detect.tflite file
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Weird bug in labelmap
if labels[0] == '???':
    del (labels[0])

# Load the tflite model
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

# FPS calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
cap = cv2.VideoCapture('src/assets/video.mp4')
time.sleep(1)
if cap.isOpened() is False:
    raise "Error opening video file"

while cap.isOpened():
    # FPS timer
    t1 = cv2.getTickCount()
    ret, frame1 = cap.read()
    if not ret:
        break

    # Getting frame from video stream, converting from BGR to RGB and resizing in order to fit the model input shape
    # frame = frame1.copy()
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame1, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Check for floating point models <<< bad FPS
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # actual detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # bounding boxes of objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # list of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # list of object scores

    # iteration over results
    for i in range(len(scores)):
        if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):  # threshold comparation
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            color = (255, 0, 0)  # Friendly/Neutral targets are blue, Enemies are red
            if object_name == "bottle":
                center_x = int((xmax + xmin) / 2)
                servo_rotation = int(center_x / 24)  # For 1280px wide frames
                print("Found target")
                color = (0, 0, 255)

            cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), color, 2)  # boundary box drawing
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # object label + score
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)  # label out of bounds
            cv2.rectangle(frame1, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame1, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame1, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)  # FPS counter

    cv2.imshow('Object detector', frame1)  # Show frame

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(25) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
