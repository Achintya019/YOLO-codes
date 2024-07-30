from ultralytics import YOLOv10
import cv2
import os
import yaml

# Load the YOLOv10 model
model = YOLOv10('yolov10x.pt')

# Load class names from the YAML file
with open('classes.yaml', 'r') as f:
    classes_dict = yaml.safe_load(f)

# Convert the dictionary to a list of class names
classes = [classes_dict[i].strip() for i in range(len(classes_dict))]

# Print class names to debug
print("Loaded class names:", classes)

# Define the class index for "traffic light"
if "traffic light" in classes:
    traffic_light_index = classes.index("traffic light")
    print(f"Index of 'traffic light': {traffic_light_index}")
else:
    raise ValueError("'traffic light' is not in the class list")

# Ensure the output directory exists
output_path = '/home/achintya-trn0175/Desktop/training/yolov10/traffic_lights_cropped_custom_vid'
os.makedirs(output_path, exist_ok=True)

# Process the results and save annotations
results = model(source='/home/achintya-trn0175/Desktop/training/yolov10/a_extracted_frames_traffic', conf=0.5, stream=True, save=True)
count = 0

for result in results:
    image_path = result.path
    raw_img = cv2.imread(image_path)
    ori_img = raw_img.copy()
    H, W = raw_img.shape[:2]

    # Check the shape of detections to ensure it's as expected
    detections = result.boxes
    if len(detections) == 0:
        #print(f"No detections in {image_path}")
        continue
    
    #print(f"Detections for {image_path}: {detections}")
    
    # Filter detections for "traffic light" category
    traffic_light_detections = [i for i, cls in enumerate(detections.cls) if cls == traffic_light_index]
    
    if traffic_light_detections:
        # Create the annotation file
        annotation_file = os.path.splitext(image_path)[0] + ".txt"
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
                
                # Draw bounding box and label on the image
                cv2.rectangle(raw_img, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)
                cv2.putText(raw_img, f'{classes[class_id]} {obj_score:.2f}', (x1, y1_expanded - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Crop and save the detected traffic light
                crop_img = ori_img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                cv2.imwrite(f"{output_path}/{classes[class_id]}_{count}.jpg", crop_img)
                count += 1
                
                
            