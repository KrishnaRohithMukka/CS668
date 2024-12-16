# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:49:19 2024

@author: 92472
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

def main():
    # Load the YOLOv8 model with your pre-trained weights
    model = YOLO('yolov8n.pt')  # Using YOLOv8 nano variant for efficiency

    # Set training parameters
    epochs = 15
    batch = 8
    img_size = 416  # Resolution to resize images to
    learning_rate = 0.01  # Adjust as needed

    # Start training
    model.train(
        data=r"C:\Users\92472\Downloads\ANOMOALY\data.yaml",
        epochs=epochs,
        batch=batch,  # Adjust batch size as needed
        imgsz=img_size,  # Reduced image size for faster training
        lr0=learning_rate,  # Learning rate
        patience=3,  # Early stopping patience
        optimizer='SGD',  # Using SGD optimizer
        augment=True,  # Disable augmentation for faster training
        cache=False,
        mosaic=0.0,  # Disable mosaic augmentation
        device=0,
        workers=0,  # Set workers to 0 to avoid multiprocessing issues on Windows
        amp=False,  # Disable AMP (Automatic Mixed Precision)
        verbose=True
    )

    # Save the final trained model weights
    model.save('trained_yolov8n_anomaly_detector.pt')

if __name__ == "__main__":
    main()

