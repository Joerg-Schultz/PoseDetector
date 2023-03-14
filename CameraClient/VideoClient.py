import socket
import imagezmq
import cv2

video_file = "poseVideoLarge_20230313_121631.avi"
sender = imagezmq.ImageSender(connect_to="tcp://132.187.198.63:5555")  # biomedsvm
rpi_name = socket.gethostname()

width = 320
height = 180
jpeg_quality = 75  # 0 to 100, higher is better quality, 95 is cv2 default
useJPEG = True

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error opening file " + video_file)

elapsed_time = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    resized_image = cv2.resize(image, (width, height))
    if useJPEG:
        ret_code, resized_image = cv2.imencode(
            ".jpg", resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_image(elapsed_time, resized_image)

cap.release()