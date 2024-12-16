import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score

# Define paths
test_images_folder = r"D:\Semester 4\Project\ANOMALY\ANOMOALY\data_set\test\images"  # Path to test images
test_labels_folder = r"D:\Semester 4\Project\ANOMALY\ANOMOALY\data_set\test\labels"  # Path to YOLO-format test labels
model_path = r"D:\Semester 4\Project\ANOMALY\ANOMOALY\best.pt"      # Path to trained YOLO model

# Load YOLO model
model = YOLO(model_path)

# Helper function to parse YOLO labels
def parse_yolo_label(label_file, image_width, image_height):
    bboxes = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            bboxes.append((class_id, x1, y1, x2, y2))
    return bboxes

# Helper function to calculate IoU
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Evaluate model
ious = []
all_ground_truths = []
all_predictions = []

for image_file in os.listdir(test_images_folder):
    image_path = os.path.join(test_images_folder, image_file)
    label_path = os.path.join(test_labels_folder, os.path.splitext(image_file)[0] + ".txt")
    
    # Load image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    
    # Get ground truth from label file
    ground_truth_bboxes = parse_yolo_label(label_path, image_width, image_height)
    
    # Get predictions from YOLO
    results = model(image)
    predicted_bboxes = []
    for detection in results[0].boxes:
        class_id = int(detection.cls[0])
        x1, y1, x2, y2 = map(float, detection.xyxy[0])
        predicted_bboxes.append((class_id, x1, y1, x2, y2))
    
    # Calculate IoUs and match ground truth with predictions
    for gt in ground_truth_bboxes:
        best_iou = 0
        for pred in predicted_bboxes:
            if gt[0] == pred[0]:  # Match class IDs
                iou = calculate_iou(gt[1:], pred[1:])
                best_iou = max(best_iou, iou)
        ious.append(best_iou)
        all_ground_truths.append(1)
        all_predictions.append(1 if best_iou > 0.5 else 0)

# Calculate metrics
precision = precision_score(all_ground_truths, all_predictions)
recall = recall_score(all_ground_truths, all_predictions)
f1 = f1_score(all_ground_truths, all_predictions)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Mean IoU: {np.mean(ious):.2f}")


import pandas as pd

result_metrics = {
    'Metric': ['Precision', 'Recall', 'F1 Score', 'Mean IoU'],
    'Value': [precision, recall, f1, np.mean(ious)]
}

df = pd.DataFrame(result_metrics)
print(df.to_string(index=False))
from tabulate import tabulate 
print(tabulate(df, headers='keys', tablefmt='grid'))
