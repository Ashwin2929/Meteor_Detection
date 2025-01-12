#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:48:02 2024

@author: ap23710 (ashwin purushothamadhas)
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import os


@tf.keras.utils.register_keras_serializable(package="Custom")
def f1_score_segmentation(y_true, y_pred, threshold=0.5):
    """
    Compute F1 score for segmentation tasks.
    """
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)  # True Positives
    fp = tf.reduce_sum(y_pred * (1 - y_true))  # False Positives
    fn = tf.reduce_sum((1 - y_pred) * y_true)  # False Negatives
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1


class YOLODetector:
    """
    Class to perform meteor detection using a YOLO model.
    """

    def __init__(self, model_path):
        """
        Initializes the YOLO model.
        """
        self.model = YOLO(model_path)

    def detect(self, input_path):
        """
        Perform detection on an input image and return bounding boxes.
        """
        results = self.model.predict(source=input_path, save=False)
        bboxes = []
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):  # Access bbox and confidence
                x1, y1, x2, y2 = box[:4]
                bboxes.append((x1.item(), y1.item(), x2.item(), y2.item(), conf.item()))
        return bboxes


class MeteorNetPredictor:
    """
    Class to perform meteor prediction using a MeteorNet model.
    """

    def __init__(self, model_path, threshold=0.5):
        """
        Initializes the MeteorNet model with a threshold for segmentation.
        """
        self.threshold = threshold
        self.model = load_model(model_path, custom_objects={"f1_score_segmentation": f1_score_segmentation})

    def predict(self, image_path):
        """
        Predict and extract bounding boxes for meteors from an input image.
        """
        image = cv2.imread(image_path)
        orig_height, orig_width = image.shape[:2]

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
            bboxes.append((x, y, x + w, y + h, 1.0))  # Assign confidence score 1.0
        return bboxes


def merge_overlapping_bboxes(bboxes):
    """
    Merge overlapping or touching bounding boxes into a single larger bounding box.
    """

    def overlaps(box1, box2):
        """
        Check if two bounding boxes overlap, touch, or one is inside the other.
        """
        # Extract coordinates
        x1, y1, x2, y2 = box1[:4]
        x1_other, y1_other, x2_other, y2_other = box2[:4]

        # Check if boxes overlap or touch
        return not (x2 < x1_other or x2_other < x1 or y2 < y1_other or y2_other < y1)

    merged_bboxes = []
    while bboxes:
        # Start with the first box
        current_box = bboxes.pop(0)
        to_merge = [current_box]

        # Find all boxes that overlap or touch the current box
        for box in bboxes[:]:
            if overlaps(current_box, box):
                to_merge.append(box)
                bboxes.remove(box)

        # Merge all related boxes
        x1 = min([box[0] for box in to_merge])
        y1 = min([box[1] for box in to_merge])
        x2 = max([box[2] for box in to_merge])
        y2 = max([box[3] for box in to_merge])
        max_conf = max([box[4] for box in to_merge])  # Keep the highest confidence score
        merged_bboxes.append((x1, y1, x2, y2, max_conf))

    return merged_bboxes


def combine_bboxes(yolo_bboxes, meteornet_bboxes):
    """
    Combine bounding boxes from YOLO and MeteorNet by merging overlapping boxes.
    """
    all_bboxes = yolo_bboxes + meteornet_bboxes  # Combine both models' bounding boxes
    merged_bboxes = merge_overlapping_bboxes(all_bboxes)
    return merged_bboxes


def visualize_combined_bboxes(image_path, combined_bboxes):
    """
    Visualize the combined bounding boxes on the input image.
    """
    original_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_image)

    for bbox in combined_bboxes:
        x1, y1, x2, y2, conf = bbox
        draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.axis("off")
    plt.title("Meteor Detection (Merged Bounding Boxes)")
    plt.show()


if __name__ == "__main__":
    """
    Main function to detect meteors using YOLO and MeteorNet models and combine their results.
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    yolo_model_folder = os.path.join("..", "trained_models", "yolo11m_model.pt")
    yolo_model_path = os.path.normpath(os.path.join(current_dir, yolo_model_folder))
    
    meteornet_model_folder = os.path.join("..", "trained_models", "meteorNet_model.keras")
    meteornet_model_path = os.path.normpath(os.path.join(current_dir, meteornet_model_folder))
    
    images_folder = os.path.join("..", "sample_spectrograms", "RAD_BEDOUR_20241209_0420_BEOPHA_SYS001.png")
    image_path = os.path.normpath(os.path.join(current_dir, images_folder))
    
    # Initialize detectors
    yolo_detector = YOLODetector(yolo_model_path)
    meteornet_predictor = MeteorNetPredictor(meteornet_model_path, threshold=0.5)

    # Get bounding boxes from both models
    yolo_bboxes = yolo_detector.detect(image_path)
    meteornet_bboxes = meteornet_predictor.predict(image_path)

    # Combine and visualize
    combined_bboxes = combine_bboxes(yolo_bboxes, meteornet_bboxes)
    visualize_combined_bboxes(image_path, combined_bboxes)
