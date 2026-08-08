"""
================================================================================
 CAR -> BOTTLE NAVIGATION SYSTEM  (Oblique Laptop Camera + Homography +
 Motion-Based Heading + Ground-Plane Waypoint Obstacle Bypass)
================================================================================

WHAT CHANGED IN THIS VERSION
-------------------------------------
The old obstacle logic checked "is anything on the line between car and
bottle" in raw PIXEL space, one frame at a time, and just replied L/R/B for
that single frame with no memory. Two things went wrong with that:

  1. With an oblique camera, pixel distances don't map to real distances
     consistently -- objects near the camera look artificially close
     together -- so the "obstacle too close, back up" check fired almost
     every time something was directly ahead, and the car just reversed
     instead of steering around.
  2. Even when it did steer L/R, that was only a single-frame nudge, not an
     actual plan to go around the object -- so the car would often turn back
     toward the bottle again next frame, see the obstacle again, and repeat.

This version fixes both by working entirely in GROUND coordinates (real cm,
via the homography) and by adding a small two-state planner:

  - DIRECT mode: drive straight at the bottle (as before).
  - AVOID mode: if an obstacle sits on the car->bottle line, compute ONE
    bypass waypoint -- a point offset sideways from the obstacle by a safety
    margin, on whichever side stays inside the calibrated arena -- and steer
    toward THAT until it's reached or the direct line to the bottle clears
    early. The waypoint is picked once and held (no re-flip-flopping every
    frame).

"Too close, back off" is now a genuine emergency check using real
centimeters, not a noisy pixel heuristic, so it only fires when something
really is right next to the car.

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

# --- Arena size -- MUST match ARENA_WIDTH_CM / ARENA_HEIGHT_CM in calibrate.py
# (used to keep bypass waypoints inside the physical table area) -------------
ARENA_WIDTH_CM = 100.0
ARENA_HEIGHT_CM = 70.0

# --- Ground-plane / heading thresholds ---------------------------------------
ARRIVAL_DIST_CM = 15             # how close (real distance) counts as "arrived"
HEADING_TOLERANCE_DEG = 15       # within this angle -> go straight
HEADING_HISTORY_FRAMES = 5       # how many frames of car movement to look back over
MIN_MOVEMENT_CM = 3              # ignore jitter smaller than this when estimating heading

# --- Obstacle bypass (ground-plane waypoint routing) --------------------------
# OBSTACLE_SAFETY_RADIUS_CM: how far (real cm) an obstacle's centre may sit
# from the straight car->bottle line before it's considered "blocking".
# Roughly: half the car's width + half the obstacle's width + a margin.
OBSTACLE_SAFETY_RADIUS_CM = 20
# Extra clearance added on top of the safety radius when placing the bypass
# waypoint out to the side of the obstacle -- gives the car some real berth
# instead of just barely clipping past it.
WAYPOINT_EXTRA_MARGIN_CM = 8
# How close the car must get to the bypass waypoint before we consider it
# "reached" and drop back into normal direct-to-bottle driving.
WAYPOINT_ARRIVAL_CM = 15
# Genuine emergency distance (real cm, ground-plane) -- if any obstacle is
# closer than this to the car right now, stop/back off immediately,
# regardless of what mode we're in.
CAR_TOO_CLOSE_CM = 12
# Where along the car->bottle line (0=at car, 1=at bottle) an obstacle must
# project to for it to count as "in the way" rather than off to the side.
BLOCK_T_MIN = 0.05
BLOCK_T_MAX = 0.95

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


# ==============================================================================
# 6. GROUND-PLANE OBSTACLE GEOMETRY (replaces the old pixel-space line check)
# ==============================================================================

def project_point_on_segment(a, b, p):
    """
    Project point p onto the segment a->b.
    Returns (t, perp_dist):
      t         -- 0.0 at a, 1.0 at b (can be <0 or >1 if p projects outside)
      perp_dist -- perpendicular distance from p to the infinite line a->b
    All inputs/outputs are in the same units (we use ground-plane cm).
    """
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
    """
    Look at every candidate obstacle's GROUND position (real cm) and find the
    one that is (a) roughly between the car and the bottle along the direct
    path, and (b) close enough to that path to actually block it. Returns the
    ground position (x_cm, y_cm) of the closest such obstacle, or None if the
    path is clear.
    """
    best_g = None
    best_dist_to_car = None

    for ob in obstacle_boxes:
        ob_g = pixel_to_ground(box_ground_point(ob), H)
        t, perp_dist = project_point_on_segment(car_g, bottle_g, ob_g)

        if BLOCK_T_MIN <= t <= BLOCK_T_MAX and perp_dist <= OBSTACLE_SAFETY_RADIUS_CM:
            dist_to_car = math.hypot(ob_g[0] - car_g[0], ob_g[1] - car_g[1])
            if best_g is None or dist_to_car < best_dist_to_car:
                best_g = ob_g
                best_dist_to_car = dist_to_car

    return best_g


def compute_bypass_waypoint(car_g, bottle_g, ob_g):
    """
    Build a single waypoint that steers around the obstacle at ob_g: a point
    offset sideways (perpendicular to the car->bottle direction) by a safety
    margin. Tries both sides and picks whichever stays inside the calibrated
    arena and is closer to the car's current position (i.e. less of a
    detour).
    """
    dx = bottle_g[0] - car_g[0]
    dy = bottle_g[1] - car_g[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    perp = (-uy, ux)  # rotate direction vector 90 degrees

    offset = OBSTACLE_SAFETY_RADIUS_CM + WAYPOINT_EXTRA_MARGIN_CM
    cand_a = (ob_g[0] + perp[0] * offset, ob_g[1] + perp[1] * offset)
    cand_b = (ob_g[0] - perp[0] * offset, ob_g[1] - perp[1] * offset)

    margin = 5.0  # small tolerance at the arena edges

    def in_bounds(c):
        return (-margin <= c[0] <= ARENA_WIDTH_CM + margin and
                -margin <= c[1] <= ARENA_HEIGHT_CM + margin)

    candidates = [c for c in (cand_a, cand_b) if in_bounds(c)]
    if not candidates:
        # Neither side is cleanly inside the arena (e.g. obstacle right at
        # the edge) -- fall back to both and just pick the nearer one.
        candidates = [cand_a, cand_b]

    candidates.sort(key=lambda c: math.hypot(c[0] - car_g[0], c[1] - car_g[1]))
    wx = min(max(candidates[0][0], 0.0), ARENA_WIDTH_CM)
    wy = min(max(candidates[0][1], 0.0), ARENA_HEIGHT_CM)
    return (wx, wy)


def steer_toward(car_g, target_g, heading_angle, reason_prefix):
    """
    Shared steering logic: turn toward target_g using the car's estimated
    real-world heading. Used for both "drive at the bottle" and "drive at
    the bypass waypoint" -- they're the same problem, just a different
    target point.
    """
    desired_angle = angle_between(car_g, target_g)

    if heading_angle is None:
        return 'F', f"{reason_prefix}: no heading yet, nudging forward"

    diff = desired_angle - heading_angle
    diff = (diff + 180) % 360 - 180  # normalize to -180..180

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
                    heading_angle, nav_state):
    """
    car_g, bottle_g: ground-plane (cm) positions, already homography-corrected.
    heading_angle: the car's current real-world heading in degrees, or None.
    nav_state: dict with 'mode' ('DIRECT' or 'AVOID') and 'waypoint'
               (ground-plane cm tuple or None), mutated in place so the
               chosen bypass route persists across frames instead of being
               recomputed (and flip-flopped) every frame.
    Returns (command, reason_text).
    """
    if car_box is None or bottle_box is None or car_g is None or bottle_g is None:
        return 'S', "car/bottle not detected"

    # ---- 1. Have we arrived? (real-world distance, not pixel overlap) --------
    dist_to_bottle = math.hypot(bottle_g[0] - car_g[0], bottle_g[1] - car_g[1])
    if dist_to_bottle <= ARRIVAL_DIST_CM:
        nav_state['mode'] = 'DIRECT'
        nav_state['waypoint'] = None
        return 'S', f"destination reached ({dist_to_bottle:.0f}cm)"

    # ---- 2. Genuine emergency: something is right next to the car RIGHT NOW,
    #         checked in real ground-plane cm (not noisy pixel distance) -----
    for ob in obstacle_boxes:
        ob_g = pixel_to_ground(box_ground_point(ob), H_MATRIX)
        d = math.hypot(ob_g[0] - car_g[0], ob_g[1] - car_g[1])
        if d < CAR_TOO_CLOSE_CM:
            nav_state['mode'] = 'DIRECT'
            nav_state['waypoint'] = None
            return 'B', f"obstacle only {d:.0f}cm away - backing off"

    # ---- 3. Already mid-bypass? Keep heading for the committed waypoint
    #         until it's reached, or the direct path to the bottle opens up
    #         early (no need to finish the whole detour if it's already clear) --
    if nav_state.get('mode') == 'AVOID' and nav_state.get('waypoint') is not None:
        wp = nav_state['waypoint']
        dist_to_wp = math.hypot(wp[0] - car_g[0], wp[1] - car_g[1])
        path_clear_now = find_blocking_obstacle_ground(
            car_g, bottle_g, obstacle_boxes, H_MATRIX) is None

        if dist_to_wp <= WAYPOINT_ARRIVAL_CM or path_clear_now:
            nav_state['mode'] = 'DIRECT'
            nav_state['waypoint'] = None
        else:
            return steer_toward(car_g, wp, heading_angle, "bypassing obstacle")

    # ---- 4. Direct mode: is the straight path to the bottle blocked? ---------
    blocking_ob_g = find_blocking_obstacle_ground(
        car_g, bottle_g, obstacle_boxes, H_MATRIX)

    if blocking_ob_g is not None:
        waypoint = compute_bypass_waypoint(car_g, bottle_g, blocking_ob_g)
        if waypoint is not None:
            nav_state['mode'] = 'AVOID'
            nav_state['waypoint'] = waypoint
            return steer_toward(car_g, waypoint, heading_angle,
                                 "obstacle ahead - routing around")

    # ---- 5. Clear path: steer straight for the bottle ------------------------
    nav_state['mode'] = 'DIRECT'
    nav_state['waypoint'] = None
    return steer_toward(car_g, bottle_g, heading_angle, "path clear")


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
# 8. DRAWING HELPERS
# ==============================================================================

def draw_box(frame, box, color, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def ground_to_pixel_approx(car_box, car_g, target_g):
    """
    Rough pixel position for drawing a ground-plane target (e.g. the bypass
    waypoint) on screen. We don't have an inverse homography set up, so we
    just anchor it near the car's own screen position, offset in the
    direction of travel -- good enough for a debug overlay, not used for
    any navigation math.
    """
    cx, cy = box_ground_point(car_box)
    dx = target_g[0] - car_g[0]
    dy = target_g[1] - car_g[1]
    scale = 3.0  # purely visual scaling factor for the debug arrow
    return (int(cx + dx * scale), int(cy - dy * scale))


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
                car_box, bottle_box, car_g, bottle_g, obstacle_boxes,
                heading_angle, nav_state
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

            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(frame, f"CMD: {command}  ({reason})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            mode_txt = f"MODE: {nav_state['mode']}"
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