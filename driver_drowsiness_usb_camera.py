# USING USB CAMERA
import os
import sys
import time
import cv2
import numpy as np
import pygame
from ultralytics import YOLO

# Initialize pygame mixer for sound
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep.wav")  # Ensure you have this file in your working directory

# User-defined parameters
model_path = 'detect_ncnn_model'
min_thresh = 0.2                   # Lowered threshold to detect smaller/farther objects
resW, resH = 1280, 720             # Resolution
record = True
power_watts = 6.0

# Check if model file exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit()

# Load the YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize USB camera
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Set up recording
record_name = 'drowsiness_detection.mp4'
record_fps = 7
recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# FPS and detection counters
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 100
eyes_closed_frames = 0
mouth_open_frames = 0
threshold_closed_eyes = 6
threshold_yawning = 10

# Main loop
while True:
    t_frame_start = time.perf_counter()

    ret, frame = cam.read()
    if not ret:
        print('Camera read failed. Exiting.')
        break

    # Run inference with timing
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

        # Always draw bounding boxes
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{classname}: {int(conf * 100)}%', (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Trigger logic only above threshold
        if classname == 'Eyes_closed' and conf > 0.15:
            detected_eyes_closed = True
        if classname == 'Yawning' and conf > min_thresh:
            detected_yawning = True

    # Update counters
    eyes_closed_frames = min(eyes_closed_frames + 1, 30) if detected_eyes_closed else max(0, eyes_closed_frames - 1)
    mouth_open_frames = min(mouth_open_frames + 1, 30) if detected_yawning else max(0, mouth_open_frames - 1)

    # Trigger alerts
    if eyes_closed_frames >= threshold_closed_eyes and mouth_open_frames >= threshold_yawning:
        cv2.putText(frame, 'DROWSY: WAKE UP!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep_sound.play()
    elif eyes_closed_frames >= threshold_closed_eyes:
        cv2.putText(frame, 'DROWSY: Eyes Closed!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep_sound.play()
    elif mouth_open_frames >= threshold_yawning:
        cv2.putText(frame, 'DROWSY: Yawning!', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep_sound.play()

    # Show FPS and metrics
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
    cv2.imshow('Drowsiness Detection', frame)
    recorder.write(frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
recorder.release()
cam.release()
cv2.destroyAllWindows()
