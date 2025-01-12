#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:44:48 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
from ultralytics import YOLO


class YOLOTrainer:
    """
    Encapsulates training and evaluation of a YOLO model.
    """
    def __init__(self, model_name="yolo11n.pt", data_file="yolo_data.yaml", output_dir="trained_models"):
        """
        Initializes paths and configuration for YOLO training and evaluation.
        """
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_name = model_name
        self.data_file = data_file
        self.output_dir = os.path.normpath(os.path.join(self.current_dir, "..", output_dir))
        self.model_path = os.path.join(self.output_dir, model_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def train_model(self, epochs=300, patience=10, optimizer="SGD", pretrained=True, batch=16, imgsz=960,
                    lr0=0.001, cos_lr=True, single_cls=True, conf=0.3, iou=0.6, auto_augment="autoaugment",
                    scale=0.2, fliplr=0.5, flipud=0.5):
        """
        Trains the YOLO model with the specified parameters.
        """
        model = YOLO(self.model_name)
        results = model.train(
            data=self.data_file,
            epochs=epochs,
            patience=patience,
            optimizer=optimizer,
            pretrained=pretrained,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            cos_lr=cos_lr,
            single_cls=single_cls,
            conf=conf,
            iou=iou,
            auto_augment=auto_augment,
            scale=scale,
            fliplr=fliplr,
            flipud=flipud,
        )
        model.save(self.model_path)
        print(f"Model saved at: {self.model_path}")
        return results

    def evaluate_model(self, split="test"):
        """
        Evaluates the trained YOLO model on the specified dataset split.
        """
        model = YOLO(self.model_path)
        test_metrics = model.val(data=self.data_file, split=split, save_json=True)
        print("Test Set Performance:")
        print(test_metrics)
        return test_metrics


def main():
    """
    Main function to train and evaluate the model.
    """
    trainer = YOLOTrainer()
    print("Training the model...")
    trainer.train_model()
    print("Evaluating the model...")
    trainer.evaluate_model()


if __name__ == "__main__":
    main()
