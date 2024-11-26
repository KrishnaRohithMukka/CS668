# Anomaly Detection and Notification System

Krishna Rohith Mukka

* *This work was realized as part of the capstone project of the MS in Data Science at Pace University*
* **Abstract:** Effective, real-time monitoring is required to ensure home, workplace, and public safety and security. Traditional manual surveillance methods are frequently inefficient and error-prone, delaying essential event reactions. This research introduces an automated deep learning-based model for anomaly identification in surveillance video feeds, specifically trained to detect incidents such as abuse, fire accidents, vandalism, traffic accidents, and violence. Using YOLOv8n for efficient object detection and 3D CNNs for temporal video analysis, this model examines video frames to detect and annotate abnormalities with high accuracy. The key evaluation metrics are: Precision: 1.00, Recall: 0.80, F1 Score: 0.89, Mean IoU: 0.70.


* **Dataset:** In this project, we have used five datasets merged to one file, they are, Fire accidents, Abuse, Car accidents, Vandalism and Violence from [universe.roboflow.com](url) .This dataset has 18704 images.
.
  * The dataset is available here: https://drive.google.com/drive/folders/1YtxUcJATtDcZTabTVsauUyY4iF9dMC0A?usp=sharing
* **Methodology:**  A pre-trained InceptionV3 model which is trained on ImageNet dataset is selected as the base model for transfer learning. Two fully connected layers are added to classify anomalies across five classes fire accidents, abuse, car accidents, vandalism, and Violence. The video feed is processed frame-by-frame using OpenCV. For each frame, the model predicts the class and confidence score. Using twilio we created a notification system which sends notification to the user when anomaly is detected.

* **Results:** Results obtained in the project with 1-2 charts / images (to be completed at the end of the semester)
* Poster (as an image)
