"""
================================================================================
 CAR -> BOTTLE NAVIGATION SYSTEM  (Oblique Laptop Camera + Homography +
 Motion-Based Heading + Ground-Plane Waypoint Obstacle Bypass)
================================================================================
"""

import cv2
import time
import math
import os
import threading
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

# --- Arena size -- MUST match ARENA_WIDTH_CM / ARENA_HEIGHT_CM in calibrate.py
# (used to keep bypass waypoints inside the physical table area) -------------
ARENA_WIDTH_CM = 100.0
ARENA_HEIGHT_CM = 70.0

# --- Ground-plane / heading thresholds ---------------------------------------
# ARRIVAL_DIST_CM is kept only for the slowdown/creep zone approach logic
# below (SLOWDOWN_DIST_CM) -- it is NOT used to decide "arrived" anymore.
# See ARRIVAL_OVERLAP_RATIO + box_overlap_fraction() for that.
ARRIVAL_DIST_CM = 15
HEADING_TOLERANCE_DEG = 15       # within this angle -> go straight
HEADING_HISTORY_FRAMES = 5       # how many frames of car movement to look back over
MIN_MOVEMENT_CM = 3              # ignore jitter smaller than this when estimating heading

# --- Obstacle bypass (ground-plane waypoint routing) --------------------------
OBSTACLE_SAFETY_RADIUS_CM = 20
WAYPOINT_EXTRA_MARGIN_CM = 8
WAYPOINT_ARRIVAL_CM = 15
CAR_TOO_CLOSE_CM = 12
BLOCK_T_MIN = 0.05
BLOCK_T_MAX = 0.95

# ---- NEW: pixel-space arrival check ------------------------------------------
# "Arrived" now means the car's and bottle's bounding boxes actually overlap
# on screen, not "ground-plane distance estimate happened to drop below a
# threshold". ARRIVAL_OVERLAP_RATIO is the fraction of the SMALLER box's
# area that must be covered by the intersection before we call it arrived:
#   1.0  = one box is completely inside the other (very strict)
#   0.35 = "almost overlapping" -- boxes are clearly touching/overlapping
#          but don't need to fully coincide (accounts for the car and
#          bottle boxes being different sizes/shapes)
# Tune this up (closer to 1.0) if it still stops too early, or down if it
# now stops too late / bumps into the bottle.
ARRIVAL_OVERLAP_RATIO = 0.35

# --- Ultrasonic <-> vision fusion (duplex obstacle avoidance) ----------------
ULTRASONIC_PLAN_CM = 30
ULTRASONIC_FORWARD_CONE_DEG = 35
ULTRASONIC_STALE_SEC = 1.0
OBSTACLE_EDGE_MARGIN_CM = 10

# --- Command smoothing ---------------------------------------------------------
COMMAND_HOLD_FRAMES = 3          # a command must repeat this many frames in a row
                                  # before it is actually sent (removes flicker) --
                                  # applies ONLY while far from the target; see below

# --- NEW: keep-alive resend --------------------------------------------------
# Many cheap Bluetooth car receivers/Arduino sketches treat "no new byte in a
# while" as "link lost" and stop the motors on their own, even if the last
# command was "drive forward". should_transmit() used to ONLY send a command
# when it *changed* from the last one sent -- so a steady "F, F, F, F..."
# never got re-sent after the first F, and the car would coast to a stop on
# its own timeout. RESEND_INTERVAL_FRAMES forces a re-send of the current
# command periodically even if it hasn't changed.
RESEND_INTERVAL_FRAMES = 10      # re-send the same command at least this often

# --- Approach behavior (prevents overshoot near the bottle) ------------------
SLOWDOWN_DIST_CM = 45
CREEP_ON_FRAMES = 2
CREEP_OFF_FRAMES = 2

# ==============================================================================
# 2. BLUETOOTH (optional) -- CONNECT THIS FIRST, before loading YOLO or the
#    camera. YOLO model loading can take many seconds; connecting Bluetooth
#    up front means it grabs the COM port the moment the script starts,
#    instead of waiting behind a slow model load and possibly losing a race
#    against something else (Arduino Serial Monitor, a leftover test script,
#    an idle timeout on the HC-05 module) for that same port.
# ==============================================================================

bluetooth_ser = None
serial_write_lock = threading.Lock()

ultrasonic_state = {'distance_cm': None, 'last_update': 0.0}
ultrasonic_lock = threading.Lock()


def serial_reader_thread(ser):
    while True:
        try:
            raw = ser.readline()
        except Exception as e:
            print(f"[WARN] Serial read error: {e}")
            time.sleep(0.5)
            continue

        if not raw:
            continue

        try:
            line = raw.decode(errors="ignore").strip()
        except Exception:
            continue

        if not line.startswith("D,"):
            continue

        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            d = float(parts[1])
        except ValueError:
            continue

        with ultrasonic_lock:
            ultrasonic_state['distance_cm'] = d if d >= 0 else None
            ultrasonic_state['last_update'] = time.time()


def get_ultrasonic_distance_cm():
    with ultrasonic_lock:
        d = ultrasonic_state['distance_cm']
        age = time.time() - ultrasonic_state['last_update']
    if d is None or age > ULTRASONIC_STALE_SEC:
        return None
    return d


if ENABLE_BLUETOOTH:
    try:
        import serial
        print(f"[INFO] Connecting to Bluetooth on {COM_PORT} @ {BAUD_RATE} baud ...")
        bluetooth_ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # give HC-05 time to establish link
        print(f"[INFO] Bluetooth CONNECTED on {COM_PORT} @ {BAUD_RATE} baud.")
        reader = threading.Thread(target=serial_reader_thread, args=(bluetooth_ser,), daemon=True)
        reader.start()
        print("[INFO] Ultrasonic duplex reader thread started.")
    except Exception as e:
        bluetooth_ser = None
        # ---- CHANGED: print the SPECIFIC exception type + message, not just
        # a generic line. This is what actually tells you WHY it failed:
        #   PermissionError / "Access is denied"  -> another program (a test
        #       script, Arduino Serial Monitor, PuTTY, etc.) still has this
        #       COM port open. Close it and re-run.
        #   FileNotFoundError / "could not open port" -> wrong COM_PORT number,
        #       or the device isn't paired/visible right now.
        #   SerialTimeoutException -> device paired but not responding.
        print("=" * 70)
        print(f"[ERROR] Bluetooth NOT connected on {COM_PORT}.")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Exception detail: {e}")
        print("[ERROR] Common cause: another program (bluetoothtest.py, Arduino")
        print("[ERROR] Serial Monitor, another navigate.py instance, etc.) is")
        print("[ERROR] still holding this COM port open. Close it, then re-run.")
        print("[ERROR] Running in PREVIEW-ONLY mode -- NO COMMANDS WILL BE SENT.")
        print("=" * 70)
else:
    print("[INFO] ENABLE_BLUETOOTH is False -- running in preview-only mode on purpose.")

# ==============================================================================
# 3. LOAD YOLO MODEL + HOMOGRAPHY
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
# 4. CAMERA
# ==============================================================================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit("[ERROR] Cannot open laptop camera. Check CAMERA_INDEX.")
print("[INFO] Camera started.")


def send_command(cmd):
    """Send a single-character command to the Arduino, if connected."""
    if bluetooth_ser is None:
        # ---- CHANGED: previously this branch just did nothing, silently.
        # Now it prints every single time so you can never mistake
        # "command decided" for "command actually sent over the wire".
        print(f"[NOT SENT] '{cmd}' -- Bluetooth is not connected.")
        return
    try:
        with serial_write_lock:
            bluetooth_ser.write(cmd.encode())
            bluetooth_ser.flush()
        print(f"[SENT] '{cmd}' over Bluetooth")
    except Exception as e:
        print(f"[WARN] Failed to send over Bluetooth: {e}")


# ==============================================================================
# 5. DETECTION HELPERS
# ==============================================================================

def detect_car_and_bottle(frame):
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
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_overlap_fraction(a, b):
    """
    Fraction (0..1) of the SMALLER of the two boxes' area that is covered by
    their pixel-space intersection. This is the arrival test: a direct,
    on-screen check of whether the car and bottle boxes actually overlap,
    rather than trusting the homography-derived ground-plane cm distance.

    Why this instead of ground-plane distance: with an oblique camera, the
    homography's accuracy is NOT uniform across the frame -- error grows the
    further a point is from the 4 calibrated corners. That let the old
    "dist_to_bottle <= ARRIVAL_DIST_CM" check fire after just one step, long
    before the car and bottle were actually near each other on screen.
    Pixel overlap can't be fooled that way -- if the boxes aren't visibly
    touching, this returns 0, full stop.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter_area = iw * ih
    if inter_area <= 0:
        return 0.0
    smaller_area = min(box_area(a), box_area(b))
    if smaller_area <= 0:
        return 0.0
    return inter_area / smaller_area


def pixel_to_ground(point, H):
    pts = np.array([[point]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, H)
    return (float(warped[0, 0, 0]), float(warped[0, 0, 1]))


def angle_between(p_from, p_to):
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(dx, dy))


def estimate_heading(history):
    if len(history) < 2:
        return None
    p_old = history[0]
    p_new = history[-1]
    dist = math.hypot(p_new[0] - p_old[0], p_new[1] - p_old[1])
    if dist < MIN_MOVEMENT_CM:
        return None
    return angle_between(p_old, p_new)


def detect_obstacles(frame, ignore_boxes):
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


# ==============================================================================
# 6. GROUND-PLANE OBSTACLE GEOMETRY
# ==============================================================================

def project_point_on_segment(a, b, p):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return 0.0, math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    perp_dist = math.hypot(px - proj_x, py - proj_y)
    return t, perp_dist


def find_blocking_obstacle_ground(car_g, bottle_g, obstacle_boxes, H):
    best_g = None
    best_box = None
    best_dist_to_car = None

    for ob in obstacle_boxes:
        ob_g = pixel_to_ground(box_ground_point(ob), H)
        t, perp_dist = project_point_on_segment(car_g, bottle_g, ob_g)

        if BLOCK_T_MIN <= t <= BLOCK_T_MAX and perp_dist <= OBSTACLE_SAFETY_RADIUS_CM:
            dist_to_car = math.hypot(ob_g[0] - car_g[0], ob_g[1] - car_g[1])
            if best_g is None or dist_to_car < best_dist_to_car:
                best_g = ob_g
                best_box = ob
                best_dist_to_car = dist_to_car

    return best_g, best_box


def find_forward_obstacle_by_ultrasonic(car_g, heading_angle, obstacle_boxes, H, ultrasonic_cm):
    if ultrasonic_cm is None or heading_angle is None:
        return None, None

    best_box = None
    best_g = None
    best_diff = None

    for ob in obstacle_boxes:
        ob_g = pixel_to_ground(box_ground_point(ob), H)
        bearing = angle_between(car_g, ob_g)
        heading_diff = (bearing - heading_angle + 180) % 360 - 180
        if abs(heading_diff) > ULTRASONIC_FORWARD_CONE_DEG:
            continue

        vision_dist = math.hypot(ob_g[0] - car_g[0], ob_g[1] - car_g[1])
        tolerance = max(20.0, ultrasonic_cm * 0.6)
        dist_diff = abs(vision_dist - ultrasonic_cm)
        if dist_diff > tolerance:
            continue

        if best_diff is None or dist_diff < best_diff:
            best_diff = dist_diff
            best_box = ob
            best_g = ob_g

    return best_g, best_box


def obstacle_ground_edges(ob_box, H):
    x1, y1, x2, y2 = ob_box
    left_px = (float(x1), float(y2))
    right_px = (float(x2), float(y2))
    left_g = pixel_to_ground(left_px, H)
    right_g = pixel_to_ground(right_px, H)
    return left_g, right_g


def compute_bypass_waypoint(car_g, bottle_g, ob_g, ob_box=None, H=None):
    dx = bottle_g[0] - car_g[0]
    dy = bottle_g[1] - car_g[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    perp = (-uy, ux)

    if ob_box is not None and H is not None:
        left_g, right_g = obstacle_ground_edges(ob_box, H)
        half_width = math.hypot(right_g[0] - left_g[0], right_g[1] - left_g[1]) / 2.0
        offset = half_width + OBSTACLE_EDGE_MARGIN_CM
    else:
        offset = OBSTACLE_SAFETY_RADIUS_CM + WAYPOINT_EXTRA_MARGIN_CM

    cand_a = (ob_g[0] + perp[0] * offset, ob_g[1] + perp[1] * offset)
    cand_b = (ob_g[0] - perp[0] * offset, ob_g[1] - perp[1] * offset)

    margin = 5.0

    def in_bounds(c):
        return (-margin <= c[0] <= ARENA_WIDTH_CM + margin and
                -margin <= c[1] <= ARENA_HEIGHT_CM + margin)

    candidates = [c for c in (cand_a, cand_b) if in_bounds(c)]
    if not candidates:
        candidates = [cand_a, cand_b]

    candidates.sort(key=lambda c: math.hypot(c[0] - car_g[0], c[1] - car_g[1]))
    wx = min(max(candidates[0][0], 0.0), ARENA_WIDTH_CM)
    wy = min(max(candidates[0][1], 0.0), ARENA_HEIGHT_CM)
    return (wx, wy)


def steer_toward(car_g, target_g, heading_angle, reason_prefix):
    desired_angle = angle_between(car_g, target_g)

    if heading_angle is None:
        return 'F', f"{reason_prefix}: no heading yet, nudging forward"

    diff = desired_angle - heading_angle
    diff = (diff + 180) % 360 - 180

    if abs(diff) <= HEADING_TOLERANCE_DEG:
        return 'F', f"{reason_prefix}: aligned (off {diff:.0f} deg)"
    elif diff > 0:
        return 'R', f"{reason_prefix}: turn right (off {diff:.0f} deg)"
    else:
        return 'L', f"{reason_prefix}: turn left (off {diff:.0f} deg)"


# ==============================================================================
# 7. DECISION LOGIC
# ==============================================================================

def decide_command(car_box, bottle_box, car_g, bottle_g, obstacle_boxes,
                    heading_angle, nav_state, ultrasonic_cm=None):
    if car_box is None or bottle_box is None or car_g is None or bottle_g is None:
        return 'S', "car/bottle not detected"

    # ---- CHANGED: arrival is decided by PIXEL-SPACE box overlap, not
    # ground-plane cm distance. See box_overlap_fraction() for why.
    overlap_ratio = box_overlap_fraction(car_box, bottle_box)
    if overlap_ratio >= ARRIVAL_OVERLAP_RATIO:
        nav_state['mode'] = 'DIRECT'
        nav_state['waypoint'] = None
        return 'S', f"destination reached (box overlap {overlap_ratio * 100:.0f}%)"

    closest_vision_d = None
    for ob in obstacle_boxes:
        ob_g = pixel_to_ground(box_ground_point(ob), H_MATRIX)
        d = math.hypot(ob_g[0] - car_g[0], ob_g[1] - car_g[1])
        if closest_vision_d is None or d < closest_vision_d:
            closest_vision_d = d

    fused_close_d = closest_vision_d
    if ultrasonic_cm is not None and (fused_close_d is None or ultrasonic_cm < fused_close_d):
        fused_close_d = ultrasonic_cm

    if fused_close_d is not None and fused_close_d < CAR_TOO_CLOSE_CM:
        nav_state['mode'] = 'DIRECT'
        nav_state['waypoint'] = None
        return 'B', f"obstacle only {fused_close_d:.0f}cm away - backing off"

    if nav_state.get('mode') == 'AVOID' and nav_state.get('waypoint') is not None:
        wp = nav_state['waypoint']
        dist_to_wp = math.hypot(wp[0] - car_g[0], wp[1] - car_g[1])
        still_blocked_g, _ = find_blocking_obstacle_ground(
            car_g, bottle_g, obstacle_boxes, H_MATRIX)
        path_clear_now = still_blocked_g is None

        if dist_to_wp <= WAYPOINT_ARRIVAL_CM or path_clear_now:
            nav_state['mode'] = 'DIRECT'
            nav_state['waypoint'] = None
        else:
            return steer_toward(car_g, wp, heading_angle, "bypassing obstacle")

    blocking_ob_g, blocking_ob_box = find_blocking_obstacle_ground(
        car_g, bottle_g, obstacle_boxes, H_MATRIX)

    ultrasonic_reason = ""
    if blocking_ob_g is None and ultrasonic_cm is not None and ultrasonic_cm <= ULTRASONIC_PLAN_CM:
        fused_g, fused_box = find_forward_obstacle_by_ultrasonic(
            car_g, heading_angle, obstacle_boxes, H_MATRIX, ultrasonic_cm)
        if fused_g is not None:
            blocking_ob_g, blocking_ob_box = fused_g, fused_box
            ultrasonic_reason = f", ultrasonic confirms {ultrasonic_cm:.0f}cm"
        else:
            rad = math.radians(heading_angle) if heading_angle is not None else 0.0
            assumed_g = (car_g[0] + math.sin(rad) * ultrasonic_cm,
                         car_g[1] + math.cos(rad) * ultrasonic_cm)
            blocking_ob_g, blocking_ob_box = assumed_g, None
            ultrasonic_reason = f", ultrasonic-only ({ultrasonic_cm:.0f}cm, no vision match)"

    if blocking_ob_g is not None:
        waypoint = compute_bypass_waypoint(
            car_g, bottle_g, blocking_ob_g, ob_box=blocking_ob_box, H=H_MATRIX)
        if waypoint is not None:
            nav_state['mode'] = 'AVOID'
            nav_state['waypoint'] = waypoint
            return steer_toward(car_g, waypoint, heading_angle,
                                 "obstacle ahead - routing around" + ultrasonic_reason)

    nav_state['mode'] = 'DIRECT'
    nav_state['waypoint'] = None
    return steer_toward(car_g, bottle_g, heading_angle, "path clear")


def post_process_command(command, reason, dist_cm, nav_state):
    """
    ---- CHANGED ----
    This used to have its OWN, second, independent arrival check based on
    ground-plane cm distance (dist_cm <= ARRIVAL_DIST_CM) -- a duplicate of
    the check in decide_command(), using the same homography-derived
    distance that was firing too early. Now it trusts decide_command()'s
    own arrival determination (pixel-box overlap) as the single source of
    truth: if decide_command() said "S, destination reached (...)", latch
    that. There's exactly one place that decides "arrived" now, not two
    disagreeing thresholds.
    """
    if nav_state['arrived']:
        return 'S', "destination reached earlier - staying stopped"

    if command == 'S' and reason.startswith("destination reached"):
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

    ---- CHANGED ----
    The old version only ever sent a command when it *changed* from the
    last one actually sent. That means a long steady run of "F, F, F, F..."
    -- exactly the case of "drive straight at the bottle for 2 seconds" --
    only transmitted ONE byte total, then went silent. If the Arduino/RC
    receiver has any kind of no-signal timeout (very common), it stops the
    motors on its own after that timeout even though Python still thinks
    it already told the car to go. This is very likely why the car looked
    like it "saw" the bottle but never actually drove to it.

    Fix: keep the same 3-frame anti-flicker debounce for far-away driving,
    but ALSO force a resend of the current command at least once every
    RESEND_INTERVAL_FRAMES, even if it hasn't changed.
    """
    close_to_target = arrived or (dist_cm is not None and dist_cm <= SLOWDOWN_DIST_CM)

    tx_state['frames_since_send'] = tx_state.get('frames_since_send', 0) + 1

    if close_to_target:
        tx_state['pending_command'] = command
        tx_state['pending_count'] = COMMAND_HOLD_FRAMES
        if command != tx_state['last_sent_command']:
            tx_state['last_sent_command'] = command
            tx_state['frames_since_send'] = 0
            return True
        if tx_state['frames_since_send'] >= RESEND_INTERVAL_FRAMES:
            tx_state['frames_since_send'] = 0
            return True
        return False

    if command == tx_state['pending_command']:
        tx_state['pending_count'] += 1
    else:
        tx_state['pending_command'] = command
        tx_state['pending_count'] = 1

    if tx_state['pending_count'] >= COMMAND_HOLD_FRAMES:
        if command != tx_state['last_sent_command']:
            tx_state['last_sent_command'] = command
            tx_state['frames_since_send'] = 0
            return True
        if tx_state['frames_since_send'] >= RESEND_INTERVAL_FRAMES:
            tx_state['frames_since_send'] = 0
            return True

    return False


# ==============================================================================
# 8. DRAWING HELPERS
# ==============================================================================

def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def ground_to_pixel_approx(car_box, car_g, target_g):
    cx, cy = box_ground_point(car_box)
    dx = target_g[0] - car_g[0]
    dy = target_g[1] - car_g[1]
    scale = 3.0
    return (int(cx + dx * scale), int(cy - dy * scale))


def draw_arrived_banner(frame):
    h, w = frame.shape[:2]
    text = "DESTINATION REACHED"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(1.0, w / 700.0)
    thickness_outline = 6
    thickness_fill = 3
    pink = (180, 20, 255)

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness_outline)
    x = max(10, (w - text_w) // 2)
    y = max(text_h + 10, (h + text_h) // 2)

    overlay = frame.copy()
    pad = 20
    cv2.rectangle(overlay, (x - pad, y - text_h - pad),
                  (x + text_w + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness_outline, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, pink, thickness_fill, cv2.LINE_AA)


# ==============================================================================
# 9. MAIN LOOP
# ==============================================================================

def main():
    car_history = deque(maxlen=HEADING_HISTORY_FRAMES)
    nav_state = {
        'arrived': False,
        'creep_counter': 0,
        'mode': 'DIRECT',
        'waypoint': None,
    }
    tx_state = {'pending_command': None, 'pending_count': 0,
                'last_sent_command': "", 'frames_since_send': 0}
    arrived_command_sent = False

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

            ultrasonic_cm = get_ultrasonic_distance_cm()

            command, reason = decide_command(
                car_box, bottle_box, car_g, bottle_g, obstacle_boxes,
                heading_angle, nav_state, ultrasonic_cm
            )
            command, reason = post_process_command(command, reason, dist_cm, nav_state)

            if not nav_state['arrived']:
                if should_transmit(command, dist_cm, nav_state['arrived'], tx_state):
                    send_command(command)
                    print(f"[COMMAND] {command}  ({reason})")
            elif not arrived_command_sent:
                send_command('S')
                print(f"[COMMAND] S  (destination reached - no further commands will be sent)")
                arrived_command_sent = True

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
                line_color = (0, 200, 255) if nav_state['mode'] == 'AVOID' else (0, 0, 255)
                cv2.line(frame, p1, p2, line_color, 1)

            if (nav_state['mode'] == 'AVOID' and nav_state['waypoint'] is not None
                    and car_box is not None and car_g is not None):
                wp_px = ground_to_pixel_approx(car_box, car_g, nav_state['waypoint'])
                car_px = (int(box_ground_point(car_box)[0]), int(box_ground_point(car_box)[1]))
                cv2.circle(frame, wp_px, 8, (0, 200, 255), 2)
                cv2.line(frame, car_px, wp_px, (0, 200, 255), 2)
                cv2.putText(frame, "WAYPOINT", (wp_px[0] + 10, wp_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            # ---- NEW: on-screen Bluetooth status, so you can see the link
            # state without watching the console.
            bt_txt = f"BT: {'CONNECTED' if bluetooth_ser is not None else 'NOT CONNECTED'}"
            bt_color = (0, 200, 0) if bluetooth_ser is not None else (0, 0, 255)
            cv2.putText(frame, bt_txt, (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, bt_txt, (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bt_color, 1)

            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            ultra_txt = f"ULTRASONIC: {ultrasonic_cm:.0f}cm" if ultrasonic_cm is not None else "ULTRASONIC: --"
            mode_txt = f"MODE: {nav_state['mode']}   {ultra_txt}"
            cv2.putText(frame, mode_txt, (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(frame, mode_txt, (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

            if car_g is not None and bottle_g is not None:
                heading_txt = f"{heading_angle:.0f}" if heading_angle is not None else "?"
                debug_txt = (f"car=({car_g[0]:.0f},{car_g[1]:.0f})cm  "
                             f"bottle=({bottle_g[0]:.0f},{bottle_g[1]:.0f})cm  "
                             f"heading={heading_txt}deg")
                cv2.putText(frame, debug_txt, (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame, debug_txt, (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # ---- NEW: show the live box-overlap ratio so you can watch it
            # approach ARRIVAL_OVERLAP_RATIO and tune the threshold if it
            # still stops too early/late.
            if car_box is not None and bottle_box is not None:
                overlap_now = box_overlap_fraction(car_box, bottle_box)
                overlap_txt = f"OVERLAP: {overlap_now * 100:.0f}%  (arrives at {ARRIVAL_OVERLAP_RATIO * 100:.0f}%)"
                cv2.putText(frame, overlap_txt, (15, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(frame, overlap_txt, (15, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

            if nav_state['arrived']:
                draw_arrived_banner(frame)

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
