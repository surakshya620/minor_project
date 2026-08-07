"""
================================================================================
 CAR -> BOTTLE NAVIGATION SYSTEM  (Overhead Webcam + YOLO OBB + Line-of-Sight
 Obstacle Check)
================================================================================

WHAT THIS SCRIPT DOES
----------------------
1. Opens your LAPTOP CAMERA only. No other sensor, no other algorithm.
2. Uses your ALREADY TRAINED YOLO model to detect two things in every frame:
       - "car"    -> your robot
       - "bottle" -> the destination
3. Draws a straight line between the CAR and the BOTTLE (the "path").
4. Looks for ANYTHING ELSE sitting on top of that straight line (no YOLO
   training needed for this part -- pure OpenCV contour detection). If some
   object's outline crosses the line, it is treated as an OBSTACLE.
5. Decides ONE command every frame:  F (forward), B (backward),
   L (turn left), R (turn right), S (stop / arrived)
   and sends it over Bluetooth (HC-05/HC-06 serial) to your Arduino.

HOW THE CAMERA IS EXPECTED TO BE PLACED
----------------------------------------
This is written for the common college-project setup: the laptop camera is
fixed ABOVE the arena, looking straight down, so it can see the car and the
bottle at the same time. (This matches the code already in your project --
line_instruct.py / live_navigation.py.) The car itself does NOT need a
camera on it.

WHAT YOU NEED TO CHANGE BEFORE RUNNING
----------------------------------------
1. MODEL_PATH   -> path to your trained best.pt (car/bottle OBB model)
2. ENABLE_BLUETOOTH -> True only once your HC-05 is paired & wired to Arduino
3. COM_PORT     -> your Bluetooth COM port (Windows) e.g. "COM6"
                   (on Linux/Mac this looks like "/dev/rfcomm0")
If you just want to TEST the logic first (no Arduino connected yet), leave
ENABLE_BLUETOOTH = False. The script will still run, show the camera window,
and PRINT the commands on screen/console instead of sending them.

HOW TO RUN
----------
    pip install -r requirements.txt
    python navigate.py

Press "q" or ESC on the video window to quit.
================================================================================
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION -- edit this block only
# ==============================================================================

MODEL_PATH = "runs/obb/car_destination_detector/weights/best.pt"

CAMERA_INDEX = 0                 # 0 = default laptop webcam
CONFIDENCE = 0.40                # YOLO detection confidence threshold

CAR_CLASS = "car"
DEST_CLASS = "bottle"

ENABLE_BLUETOOTH = True         # set True only when Arduino/HC-05 is ready
COM_PORT = "COM5"                # change to your port
BAUD_RATE = 9600

# --- Obstacle detection (pure OpenCV, no YOLO) -------------------------------
OBSTACLE_MIN_AREA = 900          # ignore tiny noise blobs (pixels^2)
OBSTACLE_THRESHOLD = 100         # 0-255, lower = only very dark objects picked up

# --- Decision thresholds ------------------------------------------------------
CENTER_TOLERANCE_PX = 35         # how "straight ahead" the bottle must be to go F
ARRIVAL_OVERLAP_RATIO = 0.40     # how much car/bottle boxes must overlap = arrived

# --- Command smoothing ---------------------------------------------------------
COMMAND_HOLD_FRAMES = 3          # a command must repeat this many frames in a row
                                  # before it is actually sent (removes flicker)

# ==============================================================================
# 2. LOAD YOLO MODEL
# ==============================================================================

print("[INFO] Loading YOLO model ...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded.")

# ==============================================================================
# 3. CAMERA
# ==============================================================================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("[ERROR] Cannot open laptop camera. Check CAMERA_INDEX.")
print("[INFO] Camera started.")

# ==============================================================================
# 4. BLUETOOTH (optional)
# ==============================================================================

bluetooth_ser = None
if ENABLE_BLUETOOTH:
    try:
        import serial
        bluetooth_ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # give HC-05 time to establish link
        print(f"[INFO] Bluetooth connected on {COM_PORT}.")
    except Exception as e:
        bluetooth_ser = None
        print(f"[WARN] Bluetooth NOT connected ({e}). Running in preview-only mode.")


def send_command(cmd):
    """Send a single-character command to the Arduino, if connected."""
    if bluetooth_ser is not None:
        try:
            bluetooth_ser.write(cmd.encode())
        except Exception as e:
            print(f"[WARN] Failed to send over Bluetooth: {e}")


# ==============================================================================
# 5. DETECTION HELPERS
# ==============================================================================

def detect_car_and_bottle(frame):
    """Run YOLO once on the frame, return (car_box, bottle_box) as (x1,y1,x2,y2)."""
    results = model.predict(frame, conf=CONFIDENCE, verbose=False)

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


def box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def boxes_overlap(a, b):
    """True if rectangle a and rectangle b overlap at all."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def detect_obstacles(frame, ignore_boxes):
    """
    Find dark/solid blobs in the frame using plain OpenCV (NO YOLO, NO
    training). Anything that is not the car or the bottle and is big enough
    is treated as a candidate obstacle.
    `ignore_boxes` = list of boxes (car_box, bottle_box) to exclude, so the
    car/bottle themselves never get flagged as obstacles.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, OBSTACLE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacle_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < OBSTACLE_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        box = (x, y, x + w, y + h)

        # skip anything that is actually the car or the bottle
        skip = False
        for ig in ignore_boxes:
            if ig is not None and boxes_overlap(box, ig):
                skip = True
                break
        if skip:
            continue

        obstacle_boxes.append(box)

    return obstacle_boxes


def line_blocked_by(box, p1, p2, frame_shape):
    """
    True if the straight line segment p1->p2 passes through `box`.
    Uses OpenCV's built-in clipLine, which is the standard, reliable way to
    test a line segment against a rectangle.
    """
    x1, y1, x2, y2 = box
    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))  # (x, y, w, h)
    inside, _, _ = cv2.clipLine(rect, p1, p2)
    return inside


def which_side(p1, p2, point):
    """
    Cross product sign: is `point` to the LEFT or RIGHT of the directed
    line from p1 -> p2 (in image coordinates)?
    Returns positive -> point is to the right, negative -> to the left.
    """
    (x1, y1), (x2, y2) = p1, p2
    (px, py) = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# ==============================================================================
# 6. DECISION LOGIC
# ==============================================================================

def decide_command(car_box, bottle_box, obstacle_boxes):
    """
    Returns (command, reason_text)
    command is one of: 'F', 'B', 'L', 'R', 'S'
    """
    if car_box is None or bottle_box is None:
        return 'S', "car/bottle not detected"

    car_c = box_center(car_box)
    bottle_c = box_center(bottle_box)

    # ---- 1. Have we arrived? -------------------------------------------------
    x_left = max(car_box[0], bottle_box[0])
    y_top = max(car_box[1], bottle_box[1])
    x_right = min(car_box[2], bottle_box[2])
    y_bottom = min(car_box[3], bottle_box[3])

    if x_right > x_left and y_bottom > y_top:
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        car_area = max(1, (car_box[2] - car_box[0]) * (car_box[3] - car_box[1]))
        if (overlap_area / car_area) > ARRIVAL_OVERLAP_RATIO:
            return 'S', "destination reached"

    # ---- 2. Is anything sitting on the straight path? ------------------------
    blocking_obstacle = None
    for ob in obstacle_boxes:
        if line_blocked_by(ob, car_c, bottle_c, None):
            blocking_obstacle = ob
            break

    if blocking_obstacle is not None:
        ob_center = box_center(blocking_obstacle)
        side = which_side(car_c, bottle_c, ob_center)

        # obstacle very close in front of the car -> back off first
        dist_to_car = np.hypot(ob_center[0] - car_c[0], ob_center[1] - car_c[1])
        car_diag = np.hypot(car_box[2] - car_box[0], car_box[3] - car_box[1])
        if dist_to_car < car_diag:
            return 'B', "obstacle too close, backing up"

        # steer away from whichever side the obstacle is sitting on
        if side >= 0:
            return 'L', "obstacle blocking path (right side) -> steering left"
        else:
            return 'R', "obstacle blocking path (left side) -> steering right"

    # ---- 3. Clear path: steer toward the bottle -------------------------------
    dx = bottle_c[0] - car_c[0]
    if abs(dx) <= CENTER_TOLERANCE_PX:
        return 'F', "path clear, bottle straight ahead"
    elif dx > 0:
        return 'R', "path clear, bottle is to the right"
    else:
        return 'L', "path clear, bottle is to the left"


# ==============================================================================
# 7. DRAWING HELPERS
# ==============================================================================

def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ==============================================================================
# 8. MAIN LOOP
# ==============================================================================

def main():
    last_sent_command = ""
    pending_command = None
    pending_count = 0

    print("[INFO] Starting navigation loop. Press 'q' or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera. Stopping.")
                break

            car_box, bottle_box = detect_car_and_bottle(frame)

            ignore = [b for b in (car_box, bottle_box) if b is not None]
            obstacle_boxes = detect_obstacles(frame, ignore)

            command, reason = decide_command(car_box, bottle_box, obstacle_boxes)

            # --- simple debounce: only send once the same command has been
            #     decided for COMMAND_HOLD_FRAMES frames in a row -----------
            if command == pending_command:
                pending_count += 1
            else:
                pending_command = command
                pending_count = 1

            if pending_count >= COMMAND_HOLD_FRAMES and command != last_sent_command:
                send_command(command)
                last_sent_command = command
                print(f"[COMMAND] {command}  ({reason})")

            # ------------------------- drawing -------------------------------
            if car_box:
                draw_box(frame, car_box, (0, 255, 0), "CAR")
            if bottle_box:
                draw_box(frame, bottle_box, (255, 0, 0), "BOTTLE")

            for ob in obstacle_boxes:
                draw_box(frame, ob, (0, 165, 255), "OBSTACLE")

            if car_box and bottle_box:
                cv2.line(frame, box_center(car_box), box_center(bottle_box),
                          (0, 0, 255), 2)

            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Car -> Bottle Navigation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if bluetooth_ser is not None:
            bluetooth_ser.close()
        print("[INFO] Stopped cleanly.")


if __name__ == "__main__":
    main()