# USING PICAMERA
import os
import sys
import time
import cv2
import numpy as np
import pygame
from ultralytics import YOLO
from picamera2 import Picamera2

# Initialize pygame mixer
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep.wav")

# Parameters
model_path = 'detect_ncnn_model'
min_thresh = 0.2
resW, resH = 1280, 720
record_fps = 7
record_name = 'drowsiness_detection.mp4'
power_watts = 6.0

# Check model path
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or not found.')
    sys.exit()

# Load model
model = YOLO(model_path)
labels = model.names

# Initialize Picamera2
picam = Picamera2()
camera_config = picam.create_video_configuration(
    main={"format": 'XRGB8888', "size": (resW, resH)}
)
picam.configure(camera_config)
picam.start()

# Recorder
recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, (resW, resH))

# Metrics
frame_rate_buffer = []
fps_avg_len = 100
eyes_closed_frames = 0
mouth_open_frames = 0
threshold_closed_eyes = 6
threshold_yawning = 10

while True:
    t_frame_start = time.perf_counter()

    # Capture from Picamera
    frame_bgra = picam.capture_array()
    frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

    # Inference
    t_infer_start = time.perf_counter()
    results = model(frame, verbose=False)
    t_infer_end = time.perf_counter()

    inference_time = t_infer_end - t_infer_start
    energy_joules = power_watts * inference_time

    detected_eyes_closed = False
    detected_yawning = False

    for box in results[0].boxes:
        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        conf = box.conf.item()
        classidx = int(box.cls.item())
        classname = labels[classidx]

        # Draw bounding boxes
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{classname}: {int(conf * 100)}%', (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if classname.lower() == 'eyes_closed' and conf > 0.15:
            detected_eyes_closed = True
        if classname.lower() == 'yawning' and conf > min_thresh:
            detected_yawning = True

    # Update counters
    eyes_closed_frames = min(eyes_closed_frames + 1, 30) if detected_eyes_closed else max(0, eyes_closed_frames - 1)
    mouth_open_frames = min(mouth_open_frames + 1, 30) if detected_yawning else max(0, mouth_open_frames - 1)

    # Alerts
    if eyes_closed_frames >= threshold_closed_eyes and mouth_open_frames >= threshold_yawning:
        cv2.putText(frame, 'DROWSY: WAKE UP!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep_sound.play()
    elif eyes_closed_frames >= threshold_closed_eyes:
        cv2.putText(frame, 'DROWSY: Eyes Closed!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep_sound.play()
    elif mouth_open_frames >= threshold_yawning:
        cv2.putText(frame, 'DROWSY: Yawning!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep_sound.play()

    # Metrics
    t_frame_end = time.perf_counter()
    frame_rate_calc = 1 / (t_frame_end - t_frame_start)
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

    cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Inference: {inference_time:.4f}s', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Energy: {energy_joules:.2f}J', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display and save
    cv2.imshow('Drowsiness Detection (PiCam)', frame)
    recorder.write(frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

# Cleanup
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
recorder.release()
picam.stop()
cv2.destroyAllWindows()
