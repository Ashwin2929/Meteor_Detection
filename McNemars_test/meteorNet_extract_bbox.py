#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:13:19 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf


class MeteorNetPredictor:
    """
    Predicts meteor bounding boxes using a trained MeteorNet model and saves them in YOLO format.
    """
    def __init__(self, model_path, threshold=0.5):
        """
        Initializes the predictor with the model path and threshold.
        """
        self.threshold = threshold
        self.model = self.load_trained_model(model_path)

    @staticmethod
    def f1_score_segmentation(y_true, y_pred):
        """
        Computes the F1 score for segmentation predictions.
        """
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def load_trained_model(self, model_path):
        """
        Loads the trained MeteorNet model with custom F1 score metric.
        """
        return load_model(model_path, custom_objects={'f1_score_segmentation': self.f1_score_segmentation})

    def predict_and_save_bboxes(self, input_folder, output_folder):
        """
        Predicts bounding boxes for all images in the input folder and saves them in YOLO format.
        """
        os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

                # Read and preprocess the image
                image = cv2.imread(image_path)
                original_image = image.copy()
                orig_height, orig_width = original_image.shape[:2]

                image_resized = cv2.resize(image, (256, 256)) / 255.0  # Normalize image
                image_resized = np.expand_dims(image_resized, axis=0)

                # Predict segmentation mask
                prediction = self.model.predict(image_resized)[0, :, :, 0]
                prediction = (prediction > self.threshold).astype(np.uint8)

                # Rescale prediction to original size
                prediction_resized = cv2.resize(prediction, (orig_width, orig_height))

                # Find bounding boxes from contours
                contours, _ = cv2.findContours(prediction_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bboxes = [cv2.boundingRect(contour) for contour in contours]
                bboxes = sorted(bboxes, key=lambda x: x[0])  # Sort by x_min

                # Save bounding boxes in YOLO format
                with open(output_file, 'w') as f:
                    for bbox in bboxes:
                        x, y, w, h = bbox
                        x_min, y_min, x_max, y_max = x, y, x + w, y + h
                        cx = (x_min + x_max) / 2.0 / orig_width  # Normalize center x
                        cy = (y_min + y_max) / 2.0 / orig_height  # Normalize center y
                        norm_w = w / orig_width  # Normalize width
                        norm_h = h / orig_height  # Normalize height
                        f.write(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")  # Class ID 0 for meteors

        print(f"Bounding box results saved to {output_folder}")


def main():
    """
    Main function to configure paths, initialize the predictor, and run predictions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths for the MeteorNet model and input images
    models_folder = os.path.join("..", "trained_models", "meteorNet_model.keras")
    meteornet_model_path = os.path.normpath(os.path.join(current_dir, models_folder))
    
    images_folder = os.path.join("..", "sample_spectrograms")
    images_path = os.path.normpath(os.path.join(current_dir, images_folder))

    # Output folder for bounding boxes
    output_folder = "sample_images_meteorNet_bbox_extracts"

    # Initialize the predictor
    predictor = MeteorNetPredictor(meteornet_model_path, threshold=0.5)

    # Run prediction and save bounding boxes
    predictor.predict_and_save_bboxes(images_path, output_folder)


if __name__ == "__main__":
    main()
