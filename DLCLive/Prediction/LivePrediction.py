from dlclive import DLCLive, Processor
import numpy as np
import imagezmq
import cv2
import tensorflow as tf
from enum import Enum

videoWidth = 320
videoHeight = 180
videoFile = '/home/binf009/tmp/testDogVideo.avi'
poseFile = '/home/binf009/tmp/testDogPose.txt'
video_file_raw = '/home/binf009/tmp/testDogVideo_raw.avi'
position_model_path = "/home/binf009/projects/positions/TFLearning/model/2/"
cutoff = 0.7  # for tensorflow predictions
smoothing_factor = 5  # take max predictions from smoothing_factor frames
recording_framerate = 20  # adapt to transfer rate


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
        down, sit, stand = positionModel.predict(np.array([flattenedPose]), verbose=0)[0]
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
# tensorFlow
positionModel = tf.keras.models.load_model(position_model_path)
# imageZMQ
image_hub = imagezmq.ImageHub()
# DeepLabCut
dlc_proc = MyProcessor()
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

# and not the fast predictions
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
