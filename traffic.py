from ultralytics import YOLOv10
import cv2
import os
import yaml

# Load the YOLOv10 model
model = YOLOv10('/home/achintya-trn0175/Desktop/training/yolov10/yolov10n_map50_046.pt')
# model_XPrera = YOLOv10('/home/achintya-trn0175/Desktop/training/yolov10/yolov10n_map50_046.pt')

with open('classes_custom.yaml', 'r') as f:
    classes = yaml.safe_load(f)
    # Check the type and content of classes
    if isinstance(classes, dict) and 'names' in classes:
        classes = classes['names']
    classes = classes['classes']
    print("Classes loaded:", classes)
    print(f"Total number of classes: {len(classes)}")


# list of file paths
# iterate over path
#  feed a path into model pretrained
#  feed a path into model custom
#  get baxa of both output
#  draw baxa of ...
#  idx ...
#  show baxa on photo ...

# Initialize the traffic light counter
traffic_light_counter = 0

# Run the model on the specified source with streaming enabled
results = model(source='/home/achintya-trn0175/testimges', conf=0.5, stream=True, save=True)

# Iterate through the results generator
for result in results:
    # Access the boxes for the current result
    boxes = result.boxes
    
    for box in boxes:
        class_index = int(box.cls[0])  # Assuming cls is an array with class index
        # Debugging print statement
        #print("Detected class index:", class_index)
        
        # Check if class_index is within valid range
        if class_index < 0 or class_index >= len(classes):
            print("Error: class_index out of range:", class_index)
        else:
            class_name = classes[class_index]  
            #print("Class name:", class_name)
            if class_name == "trafficlight":
                traffic_light_counter += 1

print(f"Total traffic lights detected: {traffic_light_counter}")
