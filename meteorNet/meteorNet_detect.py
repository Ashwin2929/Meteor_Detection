#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 01:57:23 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class MeteorNetPredictor:
    def __init__(self, model_path, threshold=0.5):
        self.threshold = threshold
        self.model = self.load_trained_model(model_path)

    @staticmethod
    def f1_score_segmentation(y_true, y_pred):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold predictions
        tp = tf.reduce_sum(y_true * y_pred)  # True Positives
        fp = tf.reduce_sum(y_pred * (1 - y_true))  # False Positives
        fn = tf.reduce_sum((1 - y_pred) * y_true)  # False Negatives
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def load_trained_model(self, model_path):
        return load_model(model_path, custom_objects={'f1_score_segmentation': self.f1_score_segmentation})

    def predict_and_visualize_bbox(self, image_path):
        image = cv2.imread(image_path)
        original_image = image.copy()
        orig_height, orig_width = original_image.shape[:2]

        # Preprocess the image
        image_resized = cv2.resize(image, (256, 256)) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        # Predict and threshold
        prediction = self.model.predict(image_resized)[0, :, :, 0]
        prediction = (prediction > self.threshold).astype(np.uint8)

        # Rescale prediction to original size
        prediction_resized = cv2.resize(prediction, (orig_width, orig_height))

        # Find contours for bounding boxes
        contours, _ = cv2.findContours(prediction_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w, y + h))
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box on the image

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with Bounding Boxes")
        plt.axis('off')
        plt.show()

        return original_image, bboxes


# Example usage
if __name__ == "__main__":
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_folder = os.path.join("..", "trained_models", "meteorNet_model.keras")
    model_path = os.path.normpath(os.path.join(current_dir, model_folder))
    
    input_folder =  os.path.join("..", "sample_spectrograms", "RAD_BEDOUR_20241209_0420_BEOPHA_SYS001.png")
    input_file_path = os.path.normpath(os.path.join(current_dir, input_folder))
    
    model_path = model_path
    image_path = input_file_path

    predictor = MeteorNetPredictor(model_path, threshold=0.5)
    original_image, bounding_boxes = predictor.predict_and_visualize_bbox(image_path)

