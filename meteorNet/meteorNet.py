#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:09:42 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import IoU, Precision, Recall, AUC, MeanIoU
import cv2

# Check for CUDA availability
if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")


class MeteorNetTrainer:
    """
    Handles the training, evaluation, and saving of the MeteorNet model.
    """
    def __init__(self, config):
        """
        Initializes the trainer with configuration and sets up save directory.
        """
        self.config = config
        self.save_dir = self.create_save_directory()

    def create_save_directory(self, base_dir="runs/meteorNet"):
        """
        Creates a directory to save training runs and model files.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(script_dir, base_dir)
        os.makedirs(root_dir, exist_ok=True)
        run_id = len([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]) + 1
        save_dir = os.path.join(root_dir, f"train{run_id}")
        os.makedirs(save_dir)
        return save_dir

    def load_data(self, image_dir, label_dir, input_shape):
        """
        Loads and preprocesses image and label data.
        """
        images, masks = [], []
        for filename in os.listdir(image_dir):
            if filename.endswith(('.png', '.jpg')):
                # Load and resize image
                image_path = os.path.join(image_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
                images.append(resized_image)

                # Load and create mask
                label_path = os.path.join(label_dir, filename.replace('.png', '.txt').replace('.jpg', '.txt'))
                mask = np.zeros((input_shape[0], input_shape[1]), dtype=np.float32)
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            _, x_center, y_center, width, height = map(float, line.strip().split())
                            x_min = int((x_center - width / 2) * input_shape[1])
                            y_min = int((y_center - height / 2) * input_shape[0])
                            x_max = int((x_center + width / 2) * input_shape[1])
                            y_max = int((y_center + height / 2) * input_shape[0])
                            mask[y_min:y_max, x_min:x_max] = 1.0
                masks.append(mask)

        images = np.array(images) / 255.0
        masks = np.array(masks).reshape(-1, input_shape[0], input_shape[1], 1)
        return images, masks

    @staticmethod
    def f1_score_segmentation(y_true, y_pred, threshold):
        """
        Computes the F1-score for segmentation.
        """
        y_pred = tf.cast(y_pred > threshold, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred * (1 - y_true))
        fn = tf.reduce_sum((1 - y_pred) * y_true)
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

    def build_model(self, input_shape, threshold):
        """
        Builds and compiles the segmentation model.
        """
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                Precision(name="precision"),
                Recall(name="recall"),
                IoU(num_classes=2, target_class_ids=[0], name="iou"),
                MeanIoU(num_classes=2, name="mean_iou"),
                AUC(name="auc"),
                lambda y_true, y_pred: self.f1_score_segmentation(y_true, y_pred, threshold),
            ]
        )
        return model

    def train_and_evaluate(self):
        """
        Trains the model, evaluates it, and saves the results.
        """
        # Load datasets
        train_images, train_masks = self.load_data(self.config["train_image_dir"], self.config["train_label_dir"], self.config["input_shape"])
        val_images, val_masks = self.load_data(self.config["val_image_dir"], self.config["val_label_dir"], self.config["input_shape"])
        test_images, test_masks = self.load_data(self.config["test_image_dir"], self.config["test_label_dir"], self.config["input_shape"])

        # Build model
        model = self.build_model(self.config["input_shape"], self.config["threshold"])

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            train_images, train_masks,
            validation_data=(val_images, val_masks),
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            callbacks=[early_stopping]
        )

        # Save model
        model_save_path = os.path.join(self.save_dir, "meteorNet_model.keras")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}.")

        # Evaluate model
        results = model.evaluate(test_images, test_masks, verbose=1)

        # Save evaluation metrics
        self.save_metrics_to_file(results)

    def save_metrics_to_file(self, results):
        """
        Saves evaluation metrics to a text file.
        """
        metrics = ['Loss', 'Precision', 'Recall', 'IoU', 'Mean IoU', 'AUC', 'F1-Score']
        metrics_file = os.path.join(self.save_dir, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("Evaluation Metrics\n")
            for metric, result in zip(metrics, results):
                f.write(f"{metric}: {result:.4f}\n")
        print(f"Metrics saved to {metrics_file}")


def resolve_paths(config):
    """
    Replaces '${base_path}' in paths with the dynamically resolved base path.
    """
    # Get the parent directory of the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # One folder up

    # Resolve the base_path relative to the parent directory
    base_path = os.path.normpath(os.path.join(parent_dir, config.get("base_path", "").lstrip("./")))

    # Update paths in the config
    for key, value in config.items():
        if isinstance(value, str) and "${base_path}" in value:
            config[key] = value.replace("${base_path}", base_path)
    return config



if __name__ == "__main__":
    # Load configuration from YAML file
    yaml_file_path = "meteorNet_data.yaml"
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Resolve ${base_path} placeholders
    config = resolve_paths(config)

    # Initialize and run the trainer
    trainer = MeteorNetTrainer(config)
    trainer.train_and_evaluate()
