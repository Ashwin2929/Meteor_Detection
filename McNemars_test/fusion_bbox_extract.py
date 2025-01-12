#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:48:22 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
def f1_score_segmentation(y_true, y_pred, threshold=0.5):
    """
    Compute F1 score for segmentation.
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
    Detects objects using a YOLO model.
    """
    def __init__(self, model_path):
        """
        Initializes the YOLO detector with the given model path.
        """
        self.model = YOLO(model_path)

    def detect(self, input_path):
        """
        Detects objects in an image and returns bounding boxes.
        """
        results = self.model.predict(source=input_path, save=False)
        bboxes = []
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = box[:4]
                bboxes.append((x1.item(), y1.item(), x2.item(), y2.item(), conf.item()))
        return bboxes

class MeteorNetPredictor:
    """
    Predicts bounding boxes using a trained MeteorNet model.
    """
    def __init__(self, model_path, threshold=0.5):
        """
        Initializes the MeteorNet predictor with the given model and threshold.
        """
        self.threshold = threshold
        self.model = load_model(model_path, custom_objects={"f1_score_segmentation": f1_score_segmentation})

    def predict(self, image_path):
        """
        Predicts bounding boxes for an image using the MeteorNet model.
        """
        image = cv2.imread(image_path)
        orig_height, orig_width = image.shape[:2]

        # Preprocess the image
        image_resized = cv2.resize(image, (256, 256)) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        # Predict segmentation mask
        prediction = self.model.predict(image_resized)[0, :, :, 0]
        prediction = (prediction > self.threshold).astype(np.uint8)

        # Rescale mask to original image size
        prediction_resized = cv2.resize(prediction, (orig_width, orig_height))

        # Find contours to determine bounding boxes
        contours, _ = cv2.findContours(prediction_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w, y + h, 1.0))  # Confidence score is 1.0
        return bboxes

def merge_overlapping_bboxes(bboxes):
    """
    Merges overlapping bounding boxes into a single bounding box.
    """
    def overlaps(box1, box2):
        x1, y1, x2, y2 = box1[:4]
        x1_other, y1_other, x2_other, y2_other = box2[:4]
        return not (x2 < x1_other or x2_other < x1 or y2 < y1_other or y2_other < y1)

    merged_bboxes = []
    while bboxes:
        current_box = bboxes.pop(0)
        to_merge = [current_box]

        for box in bboxes[:]:
            if overlaps(current_box, box):
                to_merge.append(box)
                bboxes.remove(box)

        x1 = min([box[0] for box in to_merge])
        y1 = min([box[1] for box in to_merge])
        x2 = max([box[2] for box in to_merge])
        y2 = max([box[3] for box in to_merge])
        max_conf = max([box[4] for box in to_merge])
        merged_bboxes.append((x1, y1, x2, y2, max_conf))

    return merged_bboxes

def combine_bboxes(yolo_bboxes, meteornet_bboxes):
    """
    Combines YOLO and MeteorNet bounding boxes by merging overlaps.
    """
    all_bboxes = yolo_bboxes + meteornet_bboxes
    return merge_overlapping_bboxes(all_bboxes)

def save_bboxes_to_file(output_folder, image_filename, image_path, combined_bboxes):
    """
    Saves bounding boxes to a text file in YOLO format.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{os.path.splitext(image_filename)[0]}.txt")
    with open(output_file, "w") as f:
        for bbox in combined_bboxes:
            x1, y1, x2, y2, conf = bbox
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")

if __name__ == "__main__":
    """
    Main script to perform detection, merge results, and save bounding boxes.
    """
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_model_path = os.path.normpath(os.path.join(current_dir, "..", "trained_models", "yolo11m_model.pt"))
    meteornet_model_path = os.path.normpath(os.path.join(current_dir, "..", "trained_models", "meteorNet_model.keras"))
    images_folder = os.path.normpath(os.path.join(current_dir, "..", "sample_spectrograms"))
    output_folder = "sample_images_fusion_bbox_extracts"

    # Initialize detectors
    yolo_detector = YOLODetector(yolo_model_path)
    meteornet_predictor = MeteorNetPredictor(meteornet_model_path, threshold=0.5)

    # Process images
    for filename in os.listdir(images_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(images_folder, filename)

            # Get bounding boxes from YOLO and MeteorNet
            yolo_bboxes = yolo_detector.detect(image_path)
            meteornet_bboxes = meteornet_predictor.predict(image_path)

            # Combine and save bounding boxes
            combined_bboxes = combine_bboxes(yolo_bboxes, meteornet_bboxes)
            save_bboxes_to_file(output_folder, filename, image_path, combined_bboxes)

    print(f"Bounding boxes saved in '{output_folder}'")
