import os
import cv2
import yaml
from ultralytics import YOLOv10

# Load the YOLOv10 model
model = YOLOv10('yolov10x.pt')

# Load class names from the YAML file
with open('classes.yaml', 'r') as f:
    classes_dict = yaml.safe_load(f)

# Convert the dictionary to a list of class names
classes = [classes_dict[i].strip() for i in range(len(classes_dict))]

# Define the class index for "traffic light"
if "traffic light" in classes:
    traffic_light_index = classes.index("traffic light")
    print(f"Index of 'traffic light': {traffic_light_index}")
else:
    raise ValueError("'traffic light' is not in the class list")

# Ensure the annotations_folder exists or create it if it doesn't
annotations_folder = "/home/achintya-trn0175/Desktop/traffic_annotations"
os.makedirs(annotations_folder, exist_ok=True)

# Process the results and save annotations
results = model(source='/home/achintya-trn0175/Desktop/image_folder', conf=0.5, stream=True, save=True)   #source must be folder with images only

for result in results:
    image_path = result.path
    raw_img = cv2.imread(image_path)
    ori_img = raw_img.copy()
    H, W = raw_img.shape[:2]

    detections = result.boxes

    # Filter detections for "traffic light" category
    traffic_light_detections = [i for i, cls in enumerate(detections.cls) if cls == traffic_light_index]

    if traffic_light_detections:
        # Create the annotation file in annotations_folder
        annotation_file = os.path.join(annotations_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        with open(annotation_file, 'w') as f:
            for idx in traffic_light_detections:
                box = detections.xyxy[idx]
                class_id = int(detections.cls[idx])
                obj_score = detections.conf[idx]

                x_center = (box[0] + box[2]) / 2 / W
                y_center = (box[1] + box[3]) / 2 / H
                width = (box[2] - box[0]) / W
                height = (box[3] - box[1]) / H

                # Write to the file in YOLO format
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

                # Expand the crop region by 20% along both axes
                x1, y1 = int(box[0]), int(box[1])
                x2, y2 = int(box[2]), int(box[3])

                x_margin = int(0.2 * (x2 - x1))
                y_margin = int(0.2 * (y2 - y1))

                x1_expanded = max(0, x1 - x_margin)
                y1_expanded = max(0, y1 - y_margin)
                x2_expanded = min(W, x2 + x_margin)
                y2_expanded = min(H, y2 + y_margin)

                # Draw bounding box and label on the image (optional)
                cv2.rectangle(raw_img, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)
                cv2.putText(raw_img, f'{classes[class_id]} {obj_score:.2f}', (x1, y1_expanded - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# After processing, update class indices from 9 to 7 in annotations_folder
for annotation_file in os.listdir(annotations_folder):
    if annotation_file.endswith(".txt"):
        annotation_path = os.path.join(annotations_folder, annotation_file)

        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_index = int(parts[0])
                if class_index == 9:
                    parts[0] = '7'
                    updated_line = ' '.join(parts) + '\n'
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)

        with open(annotation_path, 'w') as f:
            f.writelines(updated_lines)

# Define directories for merging annotations
directory1 = "/home/achintya-trn0175/Desktop/traffic_annotations"  # traffic light annotations
directory2 = "/home/achintya-trn0175/Desktop/object_annotations"   # other objects annotations

# Create a new directory for merged files
output_directory = "/home/achintya-trn0175/Desktop/final_annotations"
os.makedirs(output_directory, exist_ok=True)

# Get list of files in both directories
files1 = os.listdir(directory1)
files2 = os.listdir(directory2)

# Merge annotations from directory1
for filename in files1:
    if filename.endswith(".txt"):
        file1_path = os.path.join(directory1, filename)
        file2_path = os.path.join(directory2, filename)

        with open(file1_path, 'r') as f1:
            content1 = f1.read()

        if filename in files2:
            with open(file2_path, 'r') as f2:
                content2 = f2.read()
        else:
            content2 = ""

        merged_content = content1 + content2

        merged_file_path = os.path.join(output_directory, f"{filename}_merged")
        with open(merged_file_path, 'w') as f:
            f.write(merged_content)

        if filename in files2:
            files2.remove(filename)

# Copy remaining files from directory2 to output_directory
for filename in files2:
    if filename.endswith(".txt"):
        file2_path = os.path.join(directory2, filename)
        with open(file2_path, 'r') as f2:
            content = f2.read()

        merged_file_path = os.path.join(output_directory, f"{filename}_merged")
        with open(merged_file_path, 'w') as f:
            f.write(content)
