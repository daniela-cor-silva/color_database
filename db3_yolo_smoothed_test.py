import os
import cv2
from ultralytics import YOLO
import numpy as np
import random
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="YOLO inference")
parser.add_argument("--model", type=str, required=True, help="Trained YOLO model (.pt file)")
parser.add_argument("--test_images_dir", type=str, required=True, help="Test images directory")
parser.add_argument("--output_labels_dir", type=str, required=True, help="Output labels")
parser.add_argument("--output_images_dir", type=str, required=True, help="Output images")
args = parser.parse_args()

# Load your trained model
model = YOLO(args.model)

# Define paths
test_images_dir = args.test_images_dir
output_labels_dir = args.output_labels_dir
output_images_dir = args.output_images_dir


os.makedirs(output_labels_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)


test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# A color for each class
class_colors = {
    0: (0, 0, 255),    # Red
    1: (255, 0, 0),    # Blue
    2: (0, 255, 0)     # Green
}

# Define class-specific confidence thresholds
class_conf_thresholds = {
    0: 0.6,  
    1: 0.6, 
    2: 0.3   
}

# save 10% of the images with drawn bounding boxes
random.seed(42) 
sampled_images = set(random.sample(test_images, len(test_images) // 10))

# Run inference on all images
for image_name in test_images:
    # Load image
    img_path = os.path.join(test_images_dir, image_name)
    img = cv2.imread(img_path)

    # Perform inference
    results = model(img, iou=0.45)  # Use a common IoU threshold

    # Get the boxes, confidences, and class indices
    boxes = []
    confidences = []
    class_ids = []

    # Collect all boxes and their details
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            conf = box.conf[0].item()              # Confidence score
            cls = int(box.cls[0].item())           # Class index

            # Apply class-specific confidence thresholds
            if conf >= class_conf_thresholds.get(cls, 0):  # Default to 0 (if class is undefined)
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)
                class_ids.append(cls)

    # NMS to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.2)

    # Loop through detections
    output_txt_path = os.path.join(output_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")

    with open(output_txt_path, 'w') as f:
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                cls = class_ids[i]
                f.write(f"{cls} {x1} {y1} {x2} {y2} {conf:.2f}\n")

                # Draw bounding boxes if this image is part of the sample subset
                if image_name in sampled_images:
                    color = class_colors.get(cls, (255, 255, 255))  # Default to white if class is undefined
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # confidence label
                    label = f"Class {cls}: {conf:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # If no detections, still creates file
            pass

    # Save the image with bounding boxes only if it's in the sampled set
    if image_name in sampled_images:
        cv2.imwrite(os.path.join(output_images_dir, image_name), img)
