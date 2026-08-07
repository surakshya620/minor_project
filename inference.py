import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/obb/car_destination_detector/weights/best.pt")
# Change path if needed

# results = model.predict(
#     source=0,          # webcam or image path
#     conf=0.40,         # Start at 0.40 for OBB models
#     iou=0.30,          # Prevents double boxes on the same object
#     show=True
# )

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Unable to read frame from camera. Exiting...")
            break

        # Predict
        results = model(frame)

        # Draw detections
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
except KeyboardInterrupt:
    print("Inference interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()