#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:37:03 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import streamlit as st
import cv2
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, ImageDraw
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import os


class MeteorDetectionApp:
    """
    Application for meteor detection using YOLO, MeteorNet, Hybrid, and Fusion models.
    """
    def __init__(self):
        """
        Initializes paths to trained models and sets up the hybrid model directory.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))

        models_folder = os.path.join("..", "trained_models")
        yolo_model_path = os.path.normpath(os.path.join(current_dir, models_folder, "yolo11m_model.pt"))
        meteornet_model_path = os.path.normpath(os.path.join(current_dir, models_folder, "meteorNet_model.keras"))
        
        self.YOLO_MODEL_PATH = yolo_model_path
        self.METEORNET_MODEL_PATH = meteornet_model_path
        self.HYBRID_DIR = models_folder

    @staticmethod
    @tf.keras.utils.register_keras_serializable(package="Custom")
    def f1_score_segmentation(y_true, y_pred, threshold=0.5):
        """
        Computes the F1 score for segmentation predictions.
        """
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def load_yolo_model(self):
        """
        Loads the YOLO model.
        """
        return YOLO(self.YOLO_MODEL_PATH)

    def load_meteornet_model(self):
        """
        Loads the MeteorNet model with custom metrics.
        """
        return load_model(
            self.METEORNET_MODEL_PATH,
            custom_objects={"f1_score_segmentation": self.f1_score_segmentation},
        )

    def detect_with_yolo(self, image_path, yolo_model):
        """
        Detects meteors using the YOLO model and returns bounding boxes.
        """
        results = yolo_model.predict(source=image_path, save=False)
        bboxes = []
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = box[:4]
                bboxes.append((x1.item(), y1.item(), x2.item(), y2.item(), conf.item()))
        return bboxes

    def detect_with_meteornet(self, image_path, meteornet_model):
        """
        Detects meteors using the MeteorNet model and returns bounding boxes.
        """
        image = cv2.imread(image_path)
        orig_height, orig_width = image.shape[:2]
        image_resized = cv2.resize(image, (256, 256)) / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        prediction = meteornet_model.predict(image_resized)[0, :, :, 0]
        prediction = (prediction > 0.5).astype(np.uint8)
        prediction_resized = cv2.resize(prediction, (orig_width, orig_height))

        contours, _ = cv2.findContours(prediction_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x + w, y + h, 1.0))  # Confidence score 1.0
        return bboxes

    def load_hybrid_model(self):
        """
        Loads the hybrid Random Forest and feature extractor models.
        """
        with open(f"{self.HYBRID_DIR}/random_forest_model.pkl", "rb") as hybrid_file:
            hybrid_model = pickle.load(hybrid_file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        state_dict = torch.load(f"{self.HYBRID_DIR}/feature_extractor.pt", map_location=device)
        feature_extractor.load_state_dict(state_dict)
        return hybrid_model, feature_extractor

    def detect_with_hybrid(self, image_path, hybrid_model, feature_extractor, input_shape=(256, 256), patch_size=16):
        """
        Detects meteors using the hybrid model and feature extraction.
        """
        def extract_features(patch):
            patch = cv2.resize(patch, input_shape).astype("float32") / 255.0
            patch_tensor = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

            feature_extractor.eval()
            with torch.no_grad():
                raw_features = feature_extractor(patch_tensor)
            return raw_features.cpu().numpy().flatten()

        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, input_shape)
        mask = np.ones(input_shape, dtype=np.uint8)

        for y_start in range(0, input_shape[0], patch_size):
            for x_start in range(0, input_shape[1], patch_size):
                patch = image_resized[y_start:y_start + patch_size, x_start:x_start + patch_size]
                if patch.size == patch_size * patch_size * 3:
                    features = extract_features(patch)
                    prediction = hybrid_model.predict([features])[0]
                    if prediction == 0:
                        mask[y_start:y_start + patch_size, x_start:x_start + patch_size] = 0

        contours, _ = cv2.findContours((mask == 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated_image = image_resized.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def merge_overlapping_bboxes(bboxes):
        """
        Merges overlapping bounding boxes.
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

    def combine_bboxes(self, yolo_bboxes, meteornet_bboxes):
        """
        Combines bounding boxes from YOLO and MeteorNet.
        """
        all_bboxes = yolo_bboxes + meteornet_bboxes
        return self.merge_overlapping_bboxes(all_bboxes)

    def fusion_detection(self, image_path):
        """
        Performs fusion detection by combining YOLO and MeteorNet results.
        """
        yolo_model = self.load_yolo_model()
        meteornet_model = self.load_meteornet_model()

        yolo_bboxes = self.detect_with_yolo(image_path, yolo_model)
        meteornet_bboxes = self.detect_with_meteornet(image_path, meteornet_model)

        combined_bboxes = self.combine_bboxes(yolo_bboxes, meteornet_bboxes)

        original_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_image)
        for bbox in combined_bboxes:
            x1, y1, x2, y2, _ = bbox
            draw.rectangle([x1, y1, x2, y2], outline="orange", width=3)

        return original_image

    def main(self):
        """
        Main function for the Streamlit application.
        """
        st.set_page_config(page_title="MIDAS", page_icon=":comet:")
        st.title("MIDAS Meteor Detector")
        st.sidebar.title("Options")

        model_choice = st.sidebar.selectbox("Select a Model", ["YOLO", "MeteorNet", "Hybrid", "Fusion"])
        uploaded_file = st.sidebar.file_uploader("Upload a Spectrogram Image", type=["png", "jpg", "jpeg"])

        if st.sidebar.button("Detect"):
            if uploaded_file:
                with open("temp_image.png", "wb") as f:
                    f.write(uploaded_file.read())
                image_path = "temp_image.png"

                result_image = None

                if model_choice == "YOLO":
                    yolo_model = self.load_yolo_model()
                    yolo_bboxes = self.detect_with_yolo(image_path, yolo_model)
                    original_image = Image.open(image_path).convert("RGB")
                    draw = ImageDraw.Draw(original_image)
                    for bbox in yolo_bboxes:
                        x1, y1, x2, y2, _ = bbox
                        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    result_image = original_image

                elif model_choice == "MeteorNet":
                    meteornet_model = self.load_meteornet_model()
                    meteornet_bboxes = self.detect_with_meteornet(image_path, meteornet_model)
                    original_image = Image.open(image_path).convert("RGB")
                    draw = ImageDraw.Draw(original_image)
                    for bbox in meteornet_bboxes:
                        x1, y1, x2, y2, _ = bbox
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    result_image = original_image

                elif model_choice == "Hybrid":
                    hybrid_model, feature_extractor = self.load_hybrid_model()
                    result_image = self.detect_with_hybrid(image_path, hybrid_model, feature_extractor)

                elif model_choice == "Fusion":
                    result_image = self.fusion_detection(image_path)

                if isinstance(result_image, Image.Image):
                    st.image(result_image, caption="Detection Results", use_column_width=True)
                else:
                    st.error("Unexpected result. Please check the logic.")
            else:
                st.error("Please upload an image first!")


if __name__ == "__main__":
    app = MeteorDetectionApp()
    app.main()
