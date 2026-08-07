import cv2
import math
import time
import numpy as np
import serial
from ultralytics import YOLO

# ---------------- Config ----------------
MODEL_PATH = "runs/obb/car_destination_detector/weights/best.pt"                      # trained weights (destination only)
CAMERA_INDEX = 0
BLUETOOTH_PORT = "COM6"     # <-- CHANGE THIS to your port
BAUD_RATE = 9600
CONF_THRESHOLD = 0.5
HEADING_TOLERANCE_DEG = 15                   # within this angle -> go straight
MIN_MOVEMENT_PX = 4                          # ignore jitter smaller than this
COMMAND_INTERVAL = 0.3                       # seconds between sent commands
MIN_CAR_BLOB_AREA = 300                      # ignore tiny red noise specks

CLASS_DEST = "bottle"  # class name of the destination object in your YOLO model
 

# Red wraps around the HSV hue circle (0 and 180 are both "red"), so we
# need two ranges and combine them.
RED_LOWER_1 = np.array([0, 120, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 70])
RED_UPPER_2 = np.array([180, 255, 255])

# ---------------- Setup ----------------
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: could not open webcam.")
    raise SystemExit(1)

try:
    ser = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)  # allow HC-05 link to settle
    print(f"Connected to Arduino on {BLUETOOTH_PORT}")
except Exception as e:
    print(f"WARNING: could not open serial port ({e}). Running in vision-only "
          f"preview mode; no commands will be sent.")
    ser = None

prev_car_center = None
obstacle_priority = False   # True while Arduino owns the motors
last_command_time = 0
last_command_sent = None
show_mask = False


def send_command(cmd):
    """Send a single-character command to the Arduino, throttled."""
    global last_command_time, last_command_sent
    now = time.time()
    if cmd == last_command_sent and (now - last_command_time) < COMMAND_INTERVAL:
        return
    if ser is not None:
        ser.write(cmd.encode())
    last_command_time = now
    last_command_sent = cmd


def read_arduino_status():
    """Check for an incoming status line from the Arduino (non-blocking)."""
    global obstacle_priority
    if ser is None:
        return
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        print(f"[Arduino] {line}")
        if line == "OBST":
            obstacle_priority = True
        elif line == "CLR":
            obstacle_priority = False


def find_red_car(frame):
    """Return (x, y) center of the largest red blob, or None if not found."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # clean up noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    if show_mask:
        cv2.imshow("Red Mask (press 'm' to hide)", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CAR_BLOB_AREA:
        return None, None

    x, y, w, h = cv2.boundingRect(largest)
    center = (x + w / 2, y + h / 2)
    box = (x, y, x + w, y + h)
    return center, box


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def angle_between(p_from, p_to):
    """Angle in degrees of the vector p_from->p_to, 0 = pointing up (-y)."""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    angle = math.degrees(math.atan2(dx, -dy))
    return angle  # -180..180, positive = to the right


print("Starting live navigation loop.")
print("Press 'q' to quit, 'm' to toggle the red-mask debug view.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    read_arduino_status()

    # ---- Car: red color detection ----
    car_center, car_box = find_red_car(frame)

    # ---- Destination: YOLO ----
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

dest_box = None

if results.obb is not None:

    for obb in results.obb:

        cls_id = int(obb.cls[0])
        cls_name = model.names[cls_id].lower()

        if cls_name == CLASS_DEST:

            dest_box = obb.xyxy[0].tolist()
            break

    display = frame.copy()

    if car_box:
        cv2.rectangle(display, (int(car_box[0]), int(car_box[1])),
                       (int(car_box[2]), int(car_box[3])), (0, 255, 0), 2)
        cv2.putText(display, "car", (int(car_box[0]), int(car_box[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if dest_box:
        cv2.rectangle(display, (int(dest_box[0]), int(dest_box[1])),
                       (int(dest_box[2]), int(dest_box[3])), (0, 0, 255), 2)
        cv2.putText(display, "destination", (int(dest_box[0]), int(dest_box[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if car_center is not None and dest_box is not None:
        dest_center = get_center(dest_box)

        # draw the virtual straight line from car to destination
        cv2.line(display, (int(car_center[0]), int(car_center[1])),
                  (int(dest_center[0]), int(dest_center[1])), (255, 255, 0), 2)
        cv2.circle(display, (int(car_center[0]), int(car_center[1])), 5, (0, 255, 0), -1)
        cv2.circle(display, (int(dest_center[0]), int(dest_center[1])), 5, (0, 0, 255), -1)

        desired_angle = angle_between(car_center, dest_center)

        # estimate current heading from movement since last frame
        heading_angle = None
        if prev_car_center is not None:
            dist_moved = math.hypot(car_center[0] - prev_car_center[0],
                                     car_center[1] - prev_car_center[1])
            if dist_moved >= MIN_MOVEMENT_PX:
                heading_angle = angle_between(prev_car_center, car_center)

        if not obstacle_priority:
            if heading_angle is None:
                cmd = 'F'  # no reliable heading yet -> nudge forward
            else:
                diff = desired_angle - heading_angle
                diff = (diff + 180) % 360 - 180  # normalize to -180..180
                if abs(diff) <= HEADING_TOLERANCE_DEG:
                    cmd = 'F'
                elif diff > 0:
                    cmd = 'R'
                else:
                    cmd = 'L'
            send_command(cmd)
            status_text = f"CMD: {cmd}"
        else:
            status_text = "ARDUINO HAS PRIORITY (obstacle)"

        prev_car_center = car_center

        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
    else:
        # lost sight of car or destination -> stop for safety
        send_command('S')
        missing = []
        if car_center is None:
            missing.append("car")
        if dest_box is None:
            missing.append("destination")
        cv2.putText(display, f"Missing: {', '.join(missing)} - STOP", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Navigation", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        send_command('S')
        
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow("Red Mask (press 'm' to hide)")

cap.release()
cv2.destroyAllWindows()
if ser is not None:
    ser.close()
