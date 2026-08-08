"""
================================================================================
 CAMERA CALIBRATION -- run this ONCE before navigate.py
================================================================================
Your laptop camera looks ACROSS the table at an angle, not straight down.
That means a straight line in the real world does NOT look straight in the
raw camera image -- distances get compressed differently depending on how
far away they are. This script fixes that by computing a "homography": a
matrix that converts any pixel in your camera image into a real (x, y)
position on the table, in centimeters, as if you were looking straight down.

WHAT YOU NEED
-------------
Mark out a rectangle on your table (e.g. with tape or the edge of the table
itself) and know its real width and height in centimeters.

HOW TO USE
----------
1. Edit ARENA_WIDTH_CM / ARENA_HEIGHT_CM below to match your rectangle.
2. Run:  python calibrate.py
3. A frame from your camera will appear. Click the 4 corners of your
   rectangle IN THIS EXACT ORDER:
        1st click = far-left corner  (top-left as seen in the image)
        2nd click = far-right corner (top-right as seen in the image)
        3rd click = near-right corner (bottom-right, closest to camera)
        4th click = near-left corner  (bottom-left, closest to camera)
   "Far" = the edge closest to where the bottle/destination will be.
   "Near" = the edge closest to the camera / where the car starts.
4. Press 'r' any time to reset your 4 clicks and start over.
5. Once 4 points are clicked, a bird's-eye preview window pops up so you can
   confirm your rectangle now looks like an actual rectangle (not a
   trapezoid). Press any key to save and exit.

This saves "homography.npy" in the current folder. navigate.py reads that
file automatically -- keep the camera in the same position after this step,
or you'll need to re-run calibration if you move it.
================================================================================
"""

import cv2
import numpy as np
import sys

# ==============================================================================
# CONFIG -- edit these to match your real table/arena rectangle
# ==============================================================================

CAMERA_INDEX = 0
ARENA_WIDTH_CM = 100.0    # real width of your rectangle (left-right), in cm
ARENA_HEIGHT_CM = 70.0    # real depth of your rectangle (near-far), in cm
OUTPUT_FILE = "homography.npy"

# ==============================================================================

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"[INFO] Point {len(clicked_points)}: ({x}, {y})")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    print("[INFO] Click the 4 corners in order: far-left, far-right, near-right, near-left.")
    print("[INFO] Press 'r' to reset, 'q' to quit without saving.\n")

    window = "Calibration - click 4 corners"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    frame = None
    while True:
        ret, live_frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            sys.exit(1)
        frame = live_frame.copy()

        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, pt, 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i + 1), (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(clicked_points) >= 2:
            cv2.polylines(frame, [np.array(clicked_points, dtype=np.int32)],
                          isClosed=(len(clicked_points) == 4), color=(0, 200, 255), thickness=1)

        cv2.putText(frame, f"Clicked: {len(clicked_points)}/4  (r=reset, q=quit)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_points.clear()
            print("[INFO] Reset.")
        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        if len(clicked_points) == 4:
            break

    cap.release()

    src = np.array(clicked_points, dtype=np.float32)
    dst = np.array([
        [0.0, ARENA_HEIGHT_CM],                  # far-left  -> (0, H)
        [ARENA_WIDTH_CM, ARENA_HEIGHT_CM],        # far-right -> (W, H)
        [ARENA_WIDTH_CM, 0.0],                    # near-right-> (W, 0)
        [0.0, 0.0],                                # near-left -> (0, 0)
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
    np.save(OUTPUT_FILE, H)
    print(f"\n[INFO] Saved homography to {OUTPUT_FILE}")
    print("[INFO] Ground coordinate system: x = 0..%.0fcm (left-right), "
          "y = 0..%.0fcm (0=near camera, %.0f=far/near bottle side)."
          % (ARENA_WIDTH_CM, ARENA_HEIGHT_CM, ARENA_HEIGHT_CM))

    # ---- bird's-eye preview so you can sanity-check the calibration --------
    px_per_cm = 4
    warp_w = int(ARENA_WIDTH_CM * px_per_cm)
    warp_h = int(ARENA_HEIGHT_CM * px_per_cm)
    # scale dst points into pixel space for the preview only
    dst_px = dst * px_per_cm
    H_preview = cv2.getPerspectiveTransform(src, dst_px.astype(np.float32))
    warped = cv2.warpPerspective(frame, H_preview, (warp_w, warp_h))
    warped = cv2.flip(warped, 0)  # flip so "near" (y=0) shows at the bottom
    cv2.imshow("Bird's-eye preview (should look like a rectangle) - press any key",
               warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[INFO] Calibration complete. You can now run navigate.py.")


if __name__ == "__main__":
    main()