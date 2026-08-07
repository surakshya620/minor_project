
from ultralytics import YOLO

def main():
    # Start from a small pretrained YOLOv8 model (fast, good for a laptop CPU/GPU)
    model = YOLO("yolov8n-obb.pt")

    model.train(
        data=r"/Users/namunakhadka/Desktop/vision_project/DYNAMIC.v1i.yolov8-obb/data.yaml",  
        epochs=100,
        imgsz=640,
        batch=8,                     # lower to 4 if you run out of memory
        patience=20,                 # stop early if no improvement
        name="car_destination_detector"
    )

    # After training, the best weights are saved at:
    #   runs/detect/car_destination_detector/weights/best.pt
    # Copy/rename that file to "best.pt" in your project root for Step 3.

    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()

