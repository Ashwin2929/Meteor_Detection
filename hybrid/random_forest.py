#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 10:29:19 2024

@author: ap23710 (ashwin purushothamadhas)
"""


import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import time


class MeteorSegmenter:
    """
    A class to perform feature extraction, training, and segmentation for meteor detection.
    """

    def __init__(self, feature_extractor="yolo", train_image_dir=None, train_label_dir=None,
                 val_image_dir=None, val_label_dir=None, test_image_dir=None, test_label_dir=None,
                 input_shape=(256, 256), patch_size=16, log_interval=50, run_dir="runs/hybrid"):
        """
        Initializes the segmenter with parameters for feature extraction, directories, and configurations.
        """
        self.feature_extractor_type = feature_extractor
        self.train_image_dir = train_image_dir
        self.train_label_dir = train_label_dir
        self.val_image_dir = val_image_dir
        self.val_label_dir = val_label_dir
        self.test_image_dir = test_image_dir
        self.test_label_dir = test_label_dir
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_interval = log_interval
        self.run_dir = run_dir

        # Ensure the base run directory exists
        os.makedirs(self.run_dir, exist_ok=True)

        # Ensure unique run directory for this training session
        run_count = len([d for d in os.listdir(self.run_dir) if d.startswith("train")]) + 1
        self.current_run_dir = os.path.join(self.run_dir, f"train{run_count}")
        os.makedirs(self.current_run_dir, exist_ok=True)

        # Load feature extractor
        if self.feature_extractor_type == "yolo":
            self.feature_extractor = YOLO("yolov8n.pt").to(self.device)
            self.backbone = self.feature_extractor.model.model[:10]  # Adjust for YOLO backbone
        elif self.feature_extractor_type == "mobilenetv3":
            model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(self.device)
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        else:
            raise ValueError("Invalid feature_extractor type. Choose 'yolo' or 'mobilenetv3'.")
        self.backbone.eval()

    def extract_features(self, patch):
        """
        Extracts features from a given image patch using the selected feature extractor.
        """
        patch = cv2.resize(patch, self.input_shape).astype("float32") / 255.0  # Normalize
        patch = torch.tensor(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)  # Convert to CHW format
        with torch.no_grad():
            features = self.backbone(patch)
        return features.cpu().numpy().flatten()

    def load_data(self, image_dir, label_dir):
        """
        Loads and preprocesses data from the given directories for training or evaluation.
        """
        X, y = [], []
        filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
        total_files = len(filenames)
        start_time = time.time()

        for idx, filename in enumerate(filenames, 1):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, self.input_shape)

            label_path = os.path.join(label_dir, filename.replace('.png', '.txt').replace('.jpg', '.txt'))
            mask = np.ones(self.input_shape, dtype=np.uint8)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        if int(class_id) == 0:  # Meteor class
                            x_min = int((x_center - width / 2) * self.input_shape[1])
                            y_min = int((y_center - height / 2) * self.input_shape[0])
                            x_max = int((x_center + width / 2) * self.input_shape[1])
                            y_max = int((y_center + height / 2) * self.input_shape[0])
                            mask[y_min:y_max, x_min:x_max] = 0  # Meteor

            for y_start in range(0, self.input_shape[0], self.patch_size):
                for x_start in range(0, self.input_shape[1], self.patch_size):
                    patch = image[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]
                    patch_mask = mask[y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]
                    if patch.size == self.patch_size * self.patch_size * 3:
                        features = self.extract_features(patch)
                        label = 0 if np.any(patch_mask == 0) else 1
                        X.append(features)
                        y.append(label)

            # Log progress
            if idx % self.log_interval == 0 or idx == total_files:
                elapsed_time = time.time() - start_time
                avg_time_per_file = elapsed_time / idx
                remaining_time = avg_time_per_file * (total_files - idx)
                print(f"[{idx}/{total_files}] Files Processed - Elapsed: {elapsed_time:.2f}s, Remaining: {remaining_time:.2f}s")

        return np.array(X), np.array(y)

    def train_rf(self, train_X, train_y):
        """
        Trains a Random Forest classifier with the given training data.
        """
        rf_model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
        rf_model.fit(train_X, train_y)
        return rf_model

    def save_model(self, rf_model):
        """
        Saves the trained Random Forest model and the feature extractor to disk.
        """
        model_path = os.path.join(self.current_run_dir, "random_forest_model.pkl")
        with open(model_path, "wb") as model_file:
            pickle.dump(rf_model, model_file)
        print(f"Model saved to {model_path}")

        feature_extractor_path = os.path.join(self.current_run_dir, "feature_extractor.pt")
        torch.save(self.backbone.state_dict(), feature_extractor_path)
        print(f"Feature extractor saved to {feature_extractor_path}")

    def save_metrics(self, y_true, y_pred):
        """
        Saves classification metrics to a file.
        """
        report = classification_report(y_true, y_pred, target_names=["Meteor", "Background"])
        metrics_path = os.path.join(self.current_run_dir, "metrics.txt")
        with open(metrics_path, "w") as metrics_file:
            metrics_file.write(report)
        print(f"Metrics saved to {metrics_path}")


def main():
    """
    Main function to load data, train the model, and save results.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.normpath(os.path.join(current_dir, "..", "training_data", "meteors"))

    # Construct paths for train, validation, and test data
    train_image_dir = os.path.join(base_path, "train", "images")
    train_label_dir = os.path.join(base_path, "train", "labels")
    val_image_dir = os.path.join(base_path, "val", "images")
    val_label_dir = os.path.join(base_path, "val", "labels")
    test_image_dir = os.path.join(base_path, "test", "images")
    test_label_dir = os.path.join(base_path, "test", "labels")

    segmenter = MeteorSegmenter(
        feature_extractor="mobilenetv3",
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        val_image_dir=val_image_dir,
        val_label_dir=val_label_dir,
        test_image_dir=test_image_dir,
        test_label_dir=test_label_dir
    )

    print("Loading training data...")
    train_X, train_y = segmenter.load_data(segmenter.train_image_dir, segmenter.train_label_dir)

    print("Training Random Forest...")
    rf_model = segmenter.train_rf(train_X, train_y)

    print("Saving models...")
    segmenter.save_model(rf_model)

    print("Loading validation data...")
    val_X, val_y = segmenter.load_data(segmenter.val_image_dir, segmenter.val_label_dir)

    print("Evaluating on validation data...")
    val_predictions = rf_model.predict(val_X)
    segmenter.save_metrics(val_y, val_predictions)

    print(f"All outputs have been saved to: {segmenter.current_run_dir}")


if __name__ == "__main__":
    main()
