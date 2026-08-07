import cv2
import os

folder_name = "car_and_destination_dataset"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Try 0, 1, or 2 if the camera window doesn't open
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Check macOS privacy settings or change VideoCapture index.")
    exit()

count = 0

print("=== DATASET CAPTURE TOOL ===")
print("1. Place BOTH your Car and Destination in the camera field.")
print("2. Press 's' to SAVE an image.")
print("3. Move the car/destination to new spots, then press 's' again.")
print("4. Press 'q' to QUIT when you have 40-50 photos.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error: Unable to read frame from webcam.")
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved Photos: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press 'S' to Save | 'Q' to Quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Capture Dataset", display_frame)

    # Focus the pop-up window and press 's' or 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = os.path.join(folder_name, f"frame_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[{count+1}] Saved: {filename}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! All {count} images are saved in '{folder_name}'.")