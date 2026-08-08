import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "runs/obb/car_destination_detector/weights/best.pt"

COM_PORT = "COM6"          # Change if your HC-05 is on another COM port
BAUD_RATE = 9600

CAMERA_INDEX = 0

CONFIDENCE = 0.40

CAR_CLASS = "car"
DEST_CLASS = "bottle"

# Minimum contour area to consider an obstacle
MIN_OBSTACLE_AREA = 1500

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model Loaded.")

# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

print("Camera Started.")

# ==========================================================
# BLUETOOTH
# ==========================================================

try:

    ser = serial.Serial(COM_PORT, BAUD_RATE)

    time.sleep(2)

    bluetooth = True

    print("Bluetooth Connected")

except Exception as e:

    bluetooth = False

    print("Bluetooth Not Connected")

    print(e)

# ==========================================================
# SEND COMMAND
# ==========================================================

last_command = ""

def send_command(cmd):

    global last_command

    if cmd == last_command:
        return

    last_command = cmd

    print("COMMAND :", cmd)

    if bluetooth:

        ser.write(cmd.encode())

# ==========================================================
# DETECT CAR AND DESTINATION
# ==========================================================

def detect_objects(frame):

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    car_box = None
    bottle_box = None

    for r in results:

        if r.obb is None:
            continue

        for obb in r.obb:

            cls = int(obb.cls[0])

            label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, obb.xyxy[0])

            if label == CAR_CLASS:

                car_box = (x1, y1, x2, y2)

            elif label == DEST_CLASS:

                bottle_box = (x1, y1, x2, y2)

    return car_box, bottle_box

# ==========================================================
# CENTER OF BOX
# ==========================================================

def center(box):

    x1, y1, x2, y2 = box

    return (
        int((x1+x2)/2),
        int((y1+y2)/2)
    )

# ==========================================================
# DRAW BOX
# ==========================================================

def draw_box(frame, box, color, text):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        frame,
        (x1,y1),
        (x2,y2),
        color,
        2
    )

    cv2.putText(
        frame,
        text,
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# ==========================================================
# FIND OBSTACLES
# ==========================================================

def detect_obstacles(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _, thresh = cv2.threshold(
        blur,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    obstacle_boxes = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < MIN_OBSTACLE_AREA:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        obstacle_boxes.append(
            (x,y,x+w,y+h)
        )

    return obstacle_boxes