# based on Paul
# https://www.youtube.com/watch?v=fGRe4bHRoVo&list=PLGs0VKk2DiYyXlbJVaE8y1qr24YldYNDm&index=6
import socket
import time
import imagezmq
import cv2

width = 320
height = 180
recording_width = 1280
recording_height = 720
frame_rate = 20
jpeg_quality = 75  # 0 to 100, higher is better quality, 95 is cv2 default
useJPEG = True
time_string = time.strftime("%Y%m%d_%H%M%S")
file_name = f"poseVideoLarge_{time_string}.avi"
# sender = imagezmq.ImageSender(connect_to="tcp://192.168.178.44:5555")  # WLAN
# sender = imagezmq.ImageSender(connect_to="tcp://192.168.178.55:5555")  # LAN
sender = imagezmq.ImageSender(connect_to="tcp://132.187.198.63:5555")  # biomedsvm

rpi_name = socket.gethostname()
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, recording_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, recording_height)
cam.set(cv2.CAP_PROP_FPS, frame_rate)  # Can I increase this? Not limiting
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # TODO what's that doing, exactly
savedVideo = cv2.VideoWriter(file_name,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             frame_rate, (recording_width, recording_height))

tLast = time.time()
startTime = time.time()
while True:
    # read image
    ignore, image = cam.read()
    elapsedTime = time.time() - startTime
    image = cv2.flip(image, 1)
    savedVideo.write(image)
    resized_image = cv2.resize(image, (width, height))
    if useJPEG:
        ret_code, resized_image = cv2.imencode(
            ".jpg", resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_image(elapsedTime, resized_image)
    time_stamp = "%.2f" % elapsedTime
    cv2.putText(image, text=f"{time_stamp}", org=(recording_width - 120, recording_height - 20),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow("Camera", image)
    currentTime = time.time()
    fps = 1 / (currentTime - tLast)
    print(fps)
    tLast = currentTime
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

savedVideo.release()
cam.release()
