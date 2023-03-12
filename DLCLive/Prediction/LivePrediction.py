from dlclive import DLCLive, Processor
import numpy as np
import imagezmq
import cv2
import tensorflow as tf
from enum import Enum
from dvc.api import DVCFileSystem
import yaml
import mlflow.pyfunc
import sys  # TODO remove after debugging

with open("prediction_conf.yaml", 'r') as file:
    settings = yaml.safe_load(file)

data_version = settings["data_version"]
videoWidth = settings["video"]["width"]
videoHeight = settings["video"]["height"]
cutoff = settings["cutoff"]
smoothing_factor = settings["smoothing_factor"]
recording_framerate = settings["recording_framerate"]

# TODO add this to settings and add date time to file name
videoFile = '/home/binf009/tmp/testDogVideo.avi'
poseFile = '/home/binf009/tmp/testDogPose.txt'
video_file_raw = '/home/binf009/tmp/testDogVideo_raw.avi'

# get model from MLFlow registry
model_name = settings["registry_model_name"]
stage = settings["registry_model_stage"]
mlflow.set_tracking_uri("file:///home/binf009/projects/PoseDetector/DLCLive/Training/mlruns")
position_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")


class Position(Enum):
    STAND = "Stand"
    DOWN = "Down"
    SIT = "Sit"
    UNKNOWN = "Unknown"


class MyProcessor(Processor):
    def process(self, pose, **kwargs):
        global positionModel, predicted_position, smoothing_list, smoothing_factor, poseFileHandle, time_stamp
        flattenedPose = list(np.concatenate(pose).flat)
        poseString = "\t".join(list(map(str, flattenedPose)))
        down, sit, stand, unknown = positionModel.predict(np.array([flattenedPose]), verbose=0)[0]
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
# TODO change to own model.
dvc_fs = DVCFileSystem("..", rev=data_version)
dvc_file_list = dvc_fs.find("/DLCModel/exported_models", detail=False, dvc_only=True)
print(dvc_file_list)
sys.exit()
dlc_model_path = "/home/binf009/projects/ModelZoo/DLC_Dog_resnet_50_iteration-0_shuffle-0/DLC_Dog_resnet_50_iteration" \
                 "-0_shuffle-0"
dlc_live = DLCLive(dlc_model_path, processor=dlc_proc, display=True)
# OpenCV
size = (videoWidth, videoHeight)  # Can I get this from client?
result = cv2.VideoWriter(videoFile,
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
poseFileHandle = open(poseFile, "w")
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

# ~/anaconda3/envs/DLCLive_py3.7/bin/dlc-live-benchmark
# ../projects/ModelZoo/DLC_Dog_resnet_50_iteration-0_shuffle-0/DLC_Dog_resnet_50_iteration-0_shuffle-0
# testDogVideo.avi --pcutoff 0.8 --display-radius 4 --cmap bmy --save-poses --save-video -r 1
