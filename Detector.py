import cv2
import os
from ultralytics import YOLO
from twilio.rest import Client

# Twilio configuration
ACCOUNT_SID = 'ACfdd06ec4d92b6daca1cb3a2c2bafd160'     # Replace with your Twilio Account SID
AUTH_TOKEN = 'be7d0845a3bd2658be5ac63360b327a4'       # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = '+18332002494'  # Replace with your Twilio phone number
TO_PHONE_NUMBER = '+15512560922'  # Replace with recipient's phone number in E.164 format

# Initialize Twilio client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Load the trained model
model = YOLO(r"D:\Semester 4\Project\ANOMALY\ANOMOALY\best.pt")

# Define paths
video_path = r"D:\Semester 4\Project\ANOMALY\ANOMOALY\1535674-hd_1920_1080_24fps.mp4"
output_folder = 'anomaly_frames'
os.makedirs(output_folder, exist_ok=True)

# Set thresholds
confidence_threshold = 0.6
iou_threshold = 0.5

# Initialize variables
cap = cv2.VideoCapture(video_path)
frame_count = 0
anomalies_detected = False  # Flag to check if anomalies are found

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on each frame
    results = model(frame, conf=confidence_threshold, iou=iou_threshold)
    detections = results[0].boxes

    # Save frames with anomalies and set flag
    for i, detection in enumerate(detections):
        class_id = int(detection.cls[0])
        class_name = results[0].names[class_id]
        confidence = detection.conf[0]

        # Draw bounding box and label
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        label = f"{class_name} ({confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save annotated frame
        frame_path = os.path.join(output_folder, f'anomaly_{class_id}_frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)
        anomalies_detected = True  # Set flag if any anomaly is detected in this frame

    frame_count += 1

# Release video capture
cap.release()

# Send notification if anomalies were detected
if anomalies_detected:
    message = client.messages.create(
        body="Anomalies detected, check ASAP.",
        from_=TWILIO_PHONE_NUMBER,
        to=TO_PHONE_NUMBER
    )
    print(f"SMS sent: {message.sid}")

print("Processing complete. Anomaly frames saved to:", output_folder)



