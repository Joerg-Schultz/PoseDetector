import shutil

import yaml
import os
import sys
import glob
import deeplabcut

with open("prediction_conf.yaml", 'r') as file:
    settings = yaml.safe_load(file)

video_dir = "./videos" if not "video_dir" in settings else settings["video_dir"]
if not os.path.exists(video_dir):
    sys.exit("Video directory does not exist")
if not os.path.exists(f"{video_dir}/keypoints_only"):
    os.makedirs(f"{video_dir}/keypoints_only")
# DLC Model
dlc_experiment = settings["dlc_experiment"]
dlc_config = f"../../DLCModel/{dlc_experiment}/config.yaml"

video_list = glob.glob(f"{video_dir}/*.avi")
current_dir = os.getcwd()
video_list = [file.replace("./", f"{current_dir}/") for file in video_list]
deeplabcut.analyze_videos(dlc_config, video_list, save_as_csv=True)
deeplabcut.create_labeled_video(dlc_config, video_list,
                                videotype='.avi',
                                keypoints_only=True)
generated_videos = glob.glob(f"{video_dir}/*.mp4")
for video in generated_videos:
    shutil.move(video, f"{video_dir}/keypoints_only")

deeplabcut.create_labeled_video(dlc_config, video_list,
                                save_frames=True,
                                draw_skeleton=True,
                                videotype='.avi')

# remove temp directories
temp_dirs = glob.glob(f"{video_dir}/temp-*")
for dir in temp_dirs:
    shutil.rmtree(dir)

