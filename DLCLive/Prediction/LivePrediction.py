from dlclive import DLCLive, Processor
import numpy as np
import imagezmq
import cv2
from enum import Enum
import yaml
import mlflow.keras
import os
import time

with open("prediction_conf.yaml", 'r') as file:
    settings = yaml.safe_load(file)

data_version = settings["data_version"]
videoWidth = settings["video"]["width"]
videoHeight = settings["video"]["height"]
cutoff = settings["cutoff"]
smoothing_factor = settings["smoothing_factor"]
recording_framerate = settings["recording_framerate"]

video_dir = "./videos" if not "video_dir" in settings else settings["video_dir"]
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

time_string = time.strftime("%Y%m%d_%H%M%S")
video_file = f"{video_dir}/poseVideo_{time_string}.avi"
video_file_raw = f"{video_dir}/poseVideo_{time_string}_raw.avi"
pose_file = f"{video_dir}/poseData_{time_string}.txt"

# get model from MLFlow registry
model_name = settings["registry_model_name"]
stage = settings["registry_model_stage"]
mlflow.set_tracking_uri("file:///home/binf009/projects/PoseDetector/DLCLive/Training/mlruns")
position_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{stage}")

# DLC Model
dlc_experiment = settings["dlc_experiment"]
dlc_model_name = settings["dlc_model"]


class Position(Enum):
    STAND = "Stand"
    DOWN = "Down"
    SIT = "Sit"
    UNKNOWN = "Unknown"


class MyProcessor(Processor):
    def process(self, pose, **kwargs):
        global position_model, predicted_position, smoothing_list, smoothing_factor, poseFileHandle, time_stamp
        flattenedPose = list(np.concatenate(pose).flat)
        poseString = "\t".join(list(map(str, flattenedPose)))
        down, sit, stand, unknown = position_model.predict(np.array([flattenedPose]), verbose=0)[0]
        poseFileHandle.write(f"{time_stamp}\t{poseString}\t{down}\t{sit}\t{stand}\n")
        if down > cutoff:
            current_position = Position.DOWN
        elif sit > cutoff:
            current_position = Position.SIT
        elif stand > cutoff:
            current_position = Position.STAND
        else:
            current_position = Position.UNKNOWN

        if len(smoothing_list) > smoothing_factor:
            smoothing_list.pop(0)
        smoothing_list.append(current_position)
        predicted_position = max(smoothing_list, key=smoothing_list.count)
        return pose


# Setup tools
# this program
smoothing_list = []
predicted_position = Position.UNKNOWN
## tensorFlow
# positionModel = tf.keras.models.load_model(position_model_path)
# imageZMQ
image_hub = imagezmq.ImageHub()
# DeepLabCut
dlc_proc = MyProcessor()
dlc_model_path = f"../../DLCModel/{dlc_experiment}/exported-models/{dlc_model_name}"
dlc_live = DLCLive(dlc_model_path, processor=dlc_proc, display=True)
# OpenCV
size = (videoWidth, videoHeight)  # Can I get this from client?
result = cv2.VideoWriter(video_file,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         recording_framerate,
                         size)
result_raw = cv2.VideoWriter(video_file_raw,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             recording_framerate,
                             size)


def save_to_video(frame, message):
    cv2.putText(image, text=f"{message}", org=(270, 170), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    result_raw.write(image)
    if predicted_position != Position.UNKNOWN:
        cv2.putText(image, f"{predicted_position.value}", (200, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    result.write(image)


# start server and first prediction
poseFileHandle = open(pose_file, "w")
print(f"Starting with position {predicted_position.value}")
time_stamp, jpg_buffer = image_hub.recv_jpg()
image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
image_hub.send_reply(b'OK')
dlc_live.init_inference(image)
print(predicted_position.value)
save_to_video(image, predicted_position)

# and now the fast predictions
while True:
    time_stamp, jpg_buffer = image_hub.recv_jpg()
    image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
    image_hub.send_reply(b'OK')
    last_position = predicted_position
    dlc_live.get_pose(image)
    if predicted_position != last_position:
        print(predicted_position.value)
    time_stamp = "%.2f" % time_stamp
    save_to_video(image, time_stamp)

# ~/anaconda3/envs/PoseDetectorDLCLive_Prediction/bin/dlc-live-benchmark
# ../../DLCModel/DLCModel-Joerg-2023-03-09/exported-models/DLC_DLCModel_resnet_50_iteration-0_shuffle-1
# ~/tmp/testDogVideo_raw.avi --pcutoff 0.7 --display-radius 4 --cmap bmy --save-poses --save-video
# -r 1 -n 10000

