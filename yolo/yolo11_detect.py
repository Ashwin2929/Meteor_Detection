#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:12:18 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class YOLODetector:
    """
    Performs object detection and visualization using YOLO.
    """
    def __init__(self, model_path):
        """
        Initializes the YOLO model.
        """
        self.model = YOLO(model_path)

    def detect_and_display_bbox_only(self, input_path):
        """
        Detects objects and displays bounding boxes on the image.
        """
        results = self.model.predict(source=input_path, save=False)

        for result in results:
            original_image = Image.open(input_path).convert("RGB")
            draw = ImageDraw.Draw(original_image)
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box[:4]
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            plt.figure(figsize=(10, 10))
            plt.imshow(original_image)
            plt.axis("off")
            plt.title("Detections with Bounding Boxes Only")
            plt.show()


def main():
    """
    Configures paths and runs the YOLO detector.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_folder = os.path.join("..", "trained_models", "yolo11m_model.pt")
    model_path = os.path.normpath(os.path.join(current_dir, model_folder))

    input_folder = os.path.join("..", "sample_spectrograms", "RAD_BEDOUR_20241209_0420_BEOPHA_SYS001.png")
    input_file_path = os.path.normpath(os.path.join(current_dir, input_folder))

    detector = YOLODetector(model_path)
    detector.detect_and_display_bbox_only(input_file_path)


if __name__ == "__main__":
    main()
