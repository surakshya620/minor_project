"""
================================================================================
 CAR -> BOTTLE NAVIGATION SYSTEM  (Oblique Laptop Camera + Homography +
 Motion-Based Heading + Line-of-Sight Obstacle Check)
================================================================================

WHAT CHANGED FROM THE FIRST VERSION
-------------------------------------
Your laptop camera looks ACROSS the table, not straight down. That means raw
pixel positions do NOT represent real-world left/right position -- an object
close to the camera and one far away can be perfectly in line in real life
but land at different x-pixels. This version fixes that in two ways:

1. GROUND-PLANE HOMOGRAPHY. Run calibrate.py once to click the 4 corners of
   your table/arena. That produces "homography.npy", a matrix that converts
   any pixel into a real (x, y) position on the table in centimeters, as if
   the camera were mounted straight overhead. This script loads that matrix
   and uses it for every distance/direction decision.

2. MOTION-BASED HEADING. We don't know which way the car is physically
   facing just by seeing its bounding box. So the script tracks the car's
   ground position over the last few frames and computes its actual heading
   from how it has been moving -- then compares that to the direction it
   SHOULD be moving (car -> bottle) and turns to correct the difference.

Obstacle detection is UNCHANGED: plain OpenCV contour detection checks if
anything visually sits on the line between car and bottle in the camera
image. No YOLO training needed for obstacles, as before.

BEFORE RUNNING THIS
--------------------
    python calibrate.py        <- do this first, once, and don't move the camera after

HOW TO RUN
----------
    pip install -r requirements.txt
    python navigate.py

Press "q" or ESC on the video window to quit.
================================================================================
"""

import cv2
import time
import math
import os
from collections import deque

import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION -- edit this block only
# ==============================================================================

MODEL_PATH = "runs/obb/car_destination_detector/weights/best.pt"
HOMOGRAPHY_PATH = "homography.npy"

CAMERA_INDEX = 0                 # 0 = default laptop webcam
CONFIDENCE = 0.40                # YOLO detection confidence threshold

CAR_CLASS = "car"
DEST_CLASS = "bottle"

ENABLE_BLUETOOTH = True         # set True only when Arduino/HC-05 is ready
COM_PORT = "COM5"                # change to your OUTGOING bluetooth COM port
BAUD_RATE = 9600

# --- Obstacle detection (pure OpenCV, no YOLO) -------------------------------
OBSTACLE_MIN_AREA = 900          # ignore tiny noise blobs (pixels^2)
OBSTACLE_THRESHOLD = 100         # 0-255, lower = only very dark objects picked up

# --- Ground-plane / heading thresholds ---------------------------------------
ARRIVAL_DIST_CM = 15             # how close (real distance) counts as "arrived"
HEADING_TOLERANCE_DEG = 15       # within this angle -> go straight
HEADING_HISTORY_FRAMES = 5       # how many frames of car movement to look back over
MIN_MOVEMENT_CM = 3              # ignore jitter smaller than this when estimating heading

# --- Command smoothing ---------------------------------------------------------
COMMAND_HOLD_FRAMES = 3          # a command must repeat this many frames in a row
                                  # before it is actually sent (removes flicker) --
                                  # applies ONLY while far from the target; see below

# --- Approach behavior (prevents overshoot near the bottle) ------------------
# Two problems happen right near the target: (1) the arrival check can flicker
# around the threshold, so a 'stop' sometimes loses to the 3-frame debounce and
# the car just keeps rolling; (2) even a clean 'stop' can't beat the car's own
# momentum at full speed, so it coasts past the bottle. SLOWDOWN_DIST_CM fixes
# both: inside this radius the car creeps forward in short bursts (a duty
# cycle of CREEP_ON_FRAMES driving / CREEP_OFF_FRAMES stopped) instead of
# driving continuously, and every command near the target is sent immediately
# with no debounce delay, so a stop is never held up waiting to be confirmed.
SLOWDOWN_DIST_CM = 45
CREEP_ON_FRAMES = 2
CREEP_OFF_FRAMES = 2

# ==============================================================================
# 2. LOAD YOLO MODEL + HOMOGRAPHY
# ==============================================================================

print("[INFO] Loading YOLO model ...")
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded.")

if not os.path.exists(HOMOGRAPHY_PATH):
    raise SystemExit(
        f"[ERROR] '{HOMOGRAPHY_PATH}' not found.\n"
        f"        Run 'python calibrate.py' first to map your camera view to "
        f"real table coordinates, then run this script again."
    )
H_MATRIX = np.load(HOMOGRAPHY_PATH)
print(f"[INFO] Loaded homography from {HOMOGRAPHY_PATH}")

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


def box_ground_point(box):
    """
    The point of a bounding box that actually TOUCHES the table.
    For an object seen at an angle, the box's vertical CENTER is biased
    upward (toward the camera) by the object's height -- the BOTTOM edge is
    the only part of the box that reliably sits on the table surface, so
    that's what we feed into the homography.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def boxes_overlap(a, b):
    """True if rectangle a and rectangle b overlap at all."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def pixel_to_ground(point, H):
    """Convert a single (x, y) pixel into a real (x_cm, y_cm) table position."""
    pts = np.array([[point]], dtype=np.float32)   # shape (1,1,2)
    warped = cv2.perspectiveTransform(pts, H)
    return (float(warped[0, 0, 0]), float(warped[0, 0, 1]))


def angle_between(p_from, p_to):
    """
    Angle in degrees of the vector p_from -> p_to, in GROUND coordinates
    where y increases with distance away from the camera.
    0   = pointing straight ahead (away from camera)
    +90 = pointing right, -90 = pointing left
    """
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(dx, dy))


def estimate_heading(history):
    """
    Estimate the car's current real-world heading from its recent ground
    positions. Returns an angle in degrees, or None if there isn't enough
    movement yet to tell which way it's facing.
    """
    if len(history) < 2:
        return None
    p_old = history[0]
    p_new = history[-1]
    dist = math.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1])
    if dist < MIN_MOVEMENT_CM:
        return None
    return angle_between(p_old, p_new)


def detect_obstacles(frame, ignore_boxes):
    """
    Find dark/solid blobs in the frame using plain OpenCV (NO YOLO, NO
    training). Anything that is not the car or the bottle and is big enough
    is treated as a candidate obstacle.
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

        skip = False
        for ig in ignore_boxes:
            if ig is not None and boxes_overlap(box, ig):
                skip = True
                break
        if skip:
            continue

        obstacle_boxes.append(box)

    return obstacle_boxes


def line_blocked_by(box, p1, p2):
    """True if the straight line segment p1->p2 (in pixel space) passes through `box`."""
    x1, y1, x2, y2 = box
    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    p1_int = (int(round(p1[0])), int(round(p1[1])))
    p2_int = (int(round(p2[0])), int(round(p2[1])))
    inside, _, _ = cv2.clipLine(rect, p1_int, p2_int)
    return inside


def which_side(p1, p2, point):
    """Cross product sign: is `point` left or right of the directed line p1->p2."""
    (x1, y1), (x2, y2) = p1, p2
    (px, py) = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# ==============================================================================
# 6. DECISION LOGIC
# ==============================================================================

def decide_command(car_box, bottle_box, car_g, bottle_g, obstacle_boxes, heading_angle):
    """
    car_g, bottle_g: ground-plane (cm) positions, already homography-corrected.
    heading_angle: the car's current real-world heading in degrees, or None.
    Returns (command, reason_text).
    """
    if car_box is None or bottle_box is None or car_g is None or bottle_g is None:
        return 'S', "car/bottle not detected"

    # ---- 1. Have we arrived? (real-world distance, not pixel overlap) --------
    dist_cm = math.hypot(bottle_g[0] - car_g[0], bottle_g[1] - car_g[1])
    if dist_cm <= ARRIVAL_DIST_CM:
        return 'S', f"destination reached ({dist_cm:.0f}cm)"

    # ---- 2. Is anything sitting on the straight (image-space) path? ----------
    car_px = box_ground_point(car_box)
    bottle_px = box_ground_point(bottle_box)

    blocking_obstacle = None
    for ob in obstacle_boxes:
        if line_blocked_by(ob, car_px, bottle_px):
            blocking_obstacle = ob
            break

    if blocking_obstacle is not None:
        ob_center = box_center(blocking_obstacle)
        side = which_side(car_px, bottle_px, ob_center)

        dist_to_car_px = math.hypot(ob_center[0] - car_px[0], ob_center[1] - car_px[1])
        car_diag_px = math.hypot(car_box[2] - car_box[0], car_box[3] - car_box[1])
        if dist_to_car_px < car_diag_px:
            return 'B', "obstacle too close, backing up"

        if side >= 0:
            return 'L', "obstacle blocking path -> steering left"
        else:
            return 'R', "obstacle blocking path -> steering right"

    # ---- 3. Clear path: steer using real heading vs. real desired direction --
    desired_angle = angle_between(car_g, bottle_g)

    if heading_angle is None:
        return 'F', "no heading yet, nudging forward to establish direction"

    diff = desired_angle - heading_angle
    diff = (diff + 180) % 360 - 180  # normalize to -180..180

    if abs(diff) <= HEADING_TOLERANCE_DEG:
        return 'F', f"aligned (off by {diff:.0f} deg), driving forward"
    elif diff > 0:
        return 'R', f"heading off by {diff:.0f} deg -> turn right"
    else:
        return 'L', f"heading off by {diff:.0f} deg -> turn left"


def post_process_command(command, reason, dist_cm, nav_state):
    """
    Applies the arrival latch and the creep duty-cycle on top of whatever
    decide_command() said. `nav_state` is a dict with keys 'arrived' and
    'creep_counter', mutated in place so it persists across frames.
    """
    if nav_state['arrived']:
        return 'S', "destination reached earlier - staying stopped"

    if dist_cm is not None and dist_cm <= ARRIVAL_DIST_CM:
        nav_state['arrived'] = True
        return 'S', reason

    if command == 'F' and dist_cm is not None and dist_cm <= SLOWDOWN_DIST_CM:
        cycle_len = CREEP_ON_FRAMES + CREEP_OFF_FRAMES
        cycle_pos = nav_state['creep_counter'] % cycle_len
        nav_state['creep_counter'] += 1
        if cycle_pos >= CREEP_ON_FRAMES:
            return 'S', reason + " (creep: pausing)"
        else:
            return 'F', reason + " (creep: short burst)"

    nav_state['creep_counter'] = 0
    return command, reason


def should_transmit(command, dist_cm, arrived, tx_state):
    """
    Decides whether to actually call send_command() this frame.
    `tx_state` holds 'pending_command', 'pending_count', 'last_sent_command',
    mutated in place. Far from the target, a command must repeat for
    COMMAND_HOLD_FRAMES in a row before it's sent (removes flicker). Close to
    the target (creep zone or already arrived), any change is sent
    immediately -- waiting to "confirm" a stop is exactly what causes
    overshoot.
    """
    close_to_target = arrived or (dist_cm is not None and dist_cm <= SLOWDOWN_DIST_CM)

    if close_to_target:
        tx_state['pending_command'] = command
        tx_state['pending_count'] = COMMAND_HOLD_FRAMES
        if command != tx_state['last_sent_command']:
            tx_state['last_sent_command'] = command
            return True
        return False

    if command == tx_state['pending_command']:
        tx_state['pending_count'] += 1
    else:
        tx_state['pending_command'] = command
        tx_state['pending_count'] = 1

    if tx_state['pending_count'] >= COMMAND_HOLD_FRAMES and command != tx_state['last_sent_command']:
        tx_state['last_sent_command'] = command
        return True
    return False


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
    car_history = deque(maxlen=HEADING_HISTORY_FRAMES)
    nav_state = {'arrived': False, 'creep_counter': 0}
    tx_state = {'pending_command': None, 'pending_count': 0, 'last_sent_command': ""}

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

            car_g = pixel_to_ground(box_ground_point(car_box), H_MATRIX) if car_box else None
            bottle_g = pixel_to_ground(box_ground_point(bottle_box), H_MATRIX) if bottle_box else None

            if car_g is not None:
                car_history.append(car_g)
            heading_angle = estimate_heading(car_history)

            dist_cm = None
            if car_g is not None and bottle_g is not None:
                dist_cm = math.hypot(bottle_g[0] - car_g[0], bottle_g[1] - car_g[1])

            command, reason = decide_command(
                car_box, bottle_box, car_g, bottle_g, obstacle_boxes, heading_angle
            )
            command, reason = post_process_command(command, reason, dist_cm, nav_state)

            if should_transmit(command, dist_cm, nav_state['arrived'], tx_state):
                send_command(command)
                print(f"[COMMAND] {command}  ({reason})")

            # ------------------------- drawing -------------------------------
            if car_box:
                draw_box(frame, car_box, (0, 255, 0), "CAR")
            if bottle_box:
                draw_box(frame, bottle_box, (255, 0, 0), "BOTTLE")
            for ob in obstacle_boxes:
                draw_box(frame, ob, (0, 165, 255), "OBSTACLE")

            if car_box and bottle_box:
                p1 = (int(box_ground_point(car_box)[0]), int(box_ground_point(car_box)[1]))
                p2 = (int(box_ground_point(bottle_box)[0]), int(box_ground_point(bottle_box)[1]))
                cv2.line(frame, p1, p2, (0, 0, 255), 2)

            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            if car_g is not None and bottle_g is not None:
                heading_txt = f"{heading_angle:.0f}" if heading_angle is not None else "?"
                debug_txt = (f"car=({car_g[0]:.0f},{car_g[1]:.0f})cm  "
                             f"bottle=({bottle_g[0]:.0f},{bottle_g[1]:.0f})cm  "
                             f"heading={heading_txt}deg")
                cv2.putText(frame, debug_txt, (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame, debug_txt, (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("Car -> Bottle Navigation", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if bluetooth_ser is not None:
            bluetooth_ser.close()
        print("[INFO] Stopped cleanly.")


if __name__ == "__main__":
    main()