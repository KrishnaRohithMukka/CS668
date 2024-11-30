# Anomaly Detection and Notification System

Krishna Rohith Mukka

* *This work was realized as part of the capstone project of the MS in Data Science at Pace University*
* **Abstract:** Effective, real-time monitoring is required to ensure home, workplace, and public safety and security. Traditional manual surveillance methods are frequently inefficient and error-prone, delaying essential event reactions. This research introduces an automated deep learning-based model for anomaly identification in surveillance video feeds, specifically trained to detect incidents such as abuse, fire accidents, vandalism, traffic accidents, and violence. Using YOLOv8n for efficient object detection and 3D CNNs for temporal video analysis, this model examines video frames to detect and annotate abnormalities with high accuracy of 80%.


* **Dataset:** In this project, we have used five datasets merged to one file, they are, Fire accidents, Abuse, Car accidents, Vandalism and Violence from [universe.roboflow.com](url) .This dataset has 13,415 images.
.
  * The dataset is available here: https://drive.google.com/drive/folders/1YtxUcJATtDcZTabTVsauUyY4iF9dMC0A?usp=sharing
* **Methodology:**  A pre-trained InceptionV3 model which is trained on ImageNet dataset is selected as the base model for transfer learning. Two fully connected layers are added to classify anomalies across five classes fire accidents, abuse, car accidents, vandalism, and Violence. The video feed is processed frame-by-frame using OpenCV. For each frame, the model predicts the class and confidence score. Using twilio we created a notification system which sends notification to the user when anomaly is detected.

* **Results:** In this project we successfully implemented a anomaly detection system using the YOLOv8n model. The uploaded video is analyzed frame by frame and look for anomalies and detects it with higher accuracy. Frames containing detected anomalies were saved, annotated with bounding boxes and class labels. The notification system successfully sends the user notification immediately after detecting the anomalies. The Evaluation Metrics: Precision : 1.00, Recall : 0.80, F1 Score: 0.89, Mean IoU: 0.70.

* Poster (as an image)
