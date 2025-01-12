#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:47:50 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
import cv2
import numpy as np
import torch
import pickle
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from ultralytics import YOLO
import matplotlib.pyplot as plt


class MeteorTester:
    """
    A class to load a trained Random Forest model and feature extractor for testing and image segmentation.
    """

    def __init__(self, feature_extractor="yolo", model_dir="runs/random_forest/train1", input_shape=(256, 256), patch_size=16):
        """
        Initializes the tester with the required models and configurations.
        """
        self.feature_extractor_type = feature_extractor
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load saved Random Forest model
        rf_model_path = os.path.join(self.model_dir, "random_forest_model.pkl")
        with open(rf_model_path, "rb") as model_file:
            self.rf_model = pickle.load(model_file)

        # Load saved feature extractor
        feature_extractor_path = os.path.join(self.model_dir, "feature_extractor.pt")
        if self.feature_extractor_type == "yolo":
            self.feature_extractor = YOLO("yolov8n.pt").to(self.device)
            self.backbone = self.feature_extractor.model.model[:10]
        elif self.feature_extractor_type == "mobilenetv3":
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(self.device)
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError("Invalid feature_extractor type. Choose 'yolo' or 'mobilenetv3'.")

        self.backbone.load_state_dict(torch.load(feature_extractor_path, map_location=self.device))
        self.backbone.eval()

    def extract_features(self, patch):
        """
        Extracts features from an image patch using the backbone model.
        """
        patch = cv2.resize(patch, self.input_shape).astype("float32") / 255.0
        patch_tensor = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_features = self.backbone(patch_tensor)
        return raw_features.cpu().numpy().flatten()

    def segment_image(self, image):
        """
        Segments the input image and identifies bounding boxes for meteors.
        """
        mask = np.ones(self.input_shape, dtype=np.uint8)
        for y_start in range(0, self.input_shape[0], self.patch_size):
            for x_start in range(0, self.input_shape[1], self.patch_size):
                patch = image[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]
                if patch.size == self.patch_size * self.patch_size * 3:
                    features = self.extract_features(patch)
                    prediction = self.rf_model.predict([features])[0]
                    if prediction == 0:  # Meteor
                        mask[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size] = 0

        # Find bounding boxes
        contours, _ = cv2.findContours((mask == 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = [cv2.boundingRect(contour) for contour in contours]
        return bboxes

    def predict_image(self, image_path):
        """
        Performs prediction on an input image and visualizes the results with bounding boxes.
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.input_shape)
        bboxes = self.segment_image(image)

        # Draw bounding boxes on the image
        image_with_bboxes = image.copy()
        for bbox in bboxes:
            x, y, w, h = bbox
            cv2.rectangle(image_with_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image_with_bboxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction for {os.path.basename(image_path)}")
        plt.axis("off")
        plt.show()


def main():
    """
    Main function to initialize the MeteorTester and predict on a sample image.
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_folder = os.path.join("..", "trained_models")
    model_path = os.path.normpath(os.path.join(current_dir, model_folder))
    
    input_folder = os.path.join("..", "sample_spectrograms", "RAD_BEDOUR_20241209_0420_BEOPHA_SYS001.png")
    input_file_path = os.path.normpath(os.path.join(current_dir, input_folder))
    
    tester = MeteorTester(
        feature_extractor="mobilenetv3",
        model_dir=model_path,
        patch_size=16
    )

    image_path = input_file_path
    tester.predict_image(image_path)


if __name__ == "__main__":
    main()

