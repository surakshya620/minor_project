import os

# Base directory where your train, valid, test folders live
base_dir = r"/Users/suru/Desktop/vision_project/DYNAMIC.v1i.yolov8-obb"
splits = ["train", "valid", "test"]

for split in splits:
    labels_folder = os.path.join(base_dir, split, "labels")
    if not os.path.exists(labels_folder):
        continue

    print(f"Fixing label files in {split}...")
    
    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_folder, filename)
            
            with open(filepath, "r") as f:
                # Read all values, splitting by any whitespace or newlines
                tokens = f.read().split()

            if not tokens:
                continue  # Skip background images (empty label files)

            # Re-group every 9 values into 1 line (1 class_id + 8 coordinates)
            fixed_lines = []
            for i in range(0, len(tokens), 9):
                chunk = tokens[i : i + 9]
                if len(chunk) == 9:
                    # Ensure class_id is formatted as an integer (e.g. '0')
                    chunk[0] = str(int(float(chunk[0])))
                    fixed_lines.append(" ".join(chunk))

            # Overwrite the text file with single-line YOLO OBB formatting
            with open(filepath, "w") as f:
                f.write("\n".join(fixed_lines) + ("\n" if fixed_lines else ""))

print("Done! All label files have been updated.")