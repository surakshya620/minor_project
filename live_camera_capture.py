import cv2
from ultralytics import YOLO
import os

# ============================================================
# Change this path if necessary
# ============================================================
MODEL_PATH = "runs/obb/car_destination_detector/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit()

# Load trained model
model = YOLO(MODEL_PATH)

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Camera started.")
print("Press ESC to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    # Run YOLO
    results = model.predict(
        source=frame,
        conf=0.4,
        verbose=False
    )

    car_box = None
    bottle_box = None

    for r in results:

        # Skip if nothing detected
        if r.obb is None:
            continue

        for obb in r.obb:

            cls = int(obb.cls[0])
            conf = float(obb.conf[0])

            x1, y1, x2, y2 = map(int, obb.xyxy[0])

            label = model.names[cls].lower()

            # ------------------------------
            # CAR
            # ------------------------------
            if label == "car":

                car_box = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"CAR {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # ------------------------------
            # BOTTLE
            # ------------------------------
            elif label == "bottle":

                bottle_box = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)

                cv2.putText(
                    frame,
                    f"BOTTLE {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

    # ============================================================
    # Draw path
    # ============================================================
    if car_box is not None and bottle_box is not None:

        cx_car = (car_box[0] + car_box[2]) // 2
        cy_car = (car_box[1] + car_box[3]) // 2

        cx_bottle = (bottle_box[0] + bottle_box[2]) // 2
        cy_bottle = (bottle_box[1] + bottle_box[3]) // 2

        cv2.line(
            frame,
            (cx_car, cy_car),
            (cx_bottle, cy_bottle),
            (0, 0, 255),
            2,
        )

        # overlap calculation
        x_left = max(car_box[0], bottle_box[0])
        y_top = max(car_box[1], bottle_box[1])

        x_right = min(car_box[2], bottle_box[2])
        y_bottom = min(car_box[3], bottle_box[3])

        if x_right > x_left and y_bottom > y_top:

            overlap = (x_right - x_left) * (y_bottom - y_top)

            car_area = (
                (car_box[2] - car_box[0]) *
                (car_box[3] - car_box[1])
            )

            ratio = overlap / car_area

            if ratio > 0.40:

                cv2.putText(
                    frame,
                    "DESTINATION REACHED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

    # ============================================================
    # Missing detections
    # ============================================================
    if car_box is None:

        cv2.putText(
            frame,
            "CAR NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    if bottle_box is None:

        cv2.putText(
            frame,
            "BOTTLE NOT DETECTED",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Live Camera Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()