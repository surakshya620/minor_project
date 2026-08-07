import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(
"runs/obb/car_destination_detector/weights/best.pt"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    car_box = None
    bottle_box = None

    # -------------------------------------------------
    # OBB model
    # -------------------------------------------------
    for r in results:

        if r.obb is None:
            continue

        for obb in r.obb:

            cls = int(obb.cls[0])
            conf = float(obb.conf[0])

            # Convert OBB to regular rectangle
            x1, y1, x2, y2 = map(int, obb.xyxy[0])

            label = model.names[cls]

            # CAR
            if label.lower() == "car":

                car_box = (x1, y1, x2, y2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"CAR {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # BOTTLE
            elif label.lower() == "bottle":

                bottle_box = (x1, y1, x2, y2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"BOTTLE {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

    # -------------------------------------------------
    # Draw path
    # -------------------------------------------------
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
            3
        )

        # ---------------------------------------------
        # Check overlap
        # ---------------------------------------------
        x_left = max(car_box[0], bottle_box[0])
        y_top = max(car_box[1], bottle_box[1])

        x_right = min(car_box[2], bottle_box[2])
        y_bottom = min(car_box[3], bottle_box[3])

        if x_right > x_left and y_bottom > y_top:

            overlap_area = (
                (x_right - x_left)
                * (y_bottom - y_top)
            )

            car_area = (
                (car_box[2] - car_box[0])
                * (car_box[3] - car_box[1])
            )

            overlap_ratio = overlap_area / car_area

            if overlap_ratio > 0.40:

                cv2.putText(
                    frame,
                    "DESTINATION REACHED",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    # -------------------------------------------------
    # Missing detections
    # -------------------------------------------------
    if car_box is None:

        cv2.putText(
            frame,
            "CAR NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    if bottle_box is None:

        cv2.putText(
            frame,
            "BOTTLE NOT DETECTED",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Navigation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()