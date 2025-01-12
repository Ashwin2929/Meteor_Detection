#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:32:10 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
from ultralytics import YOLO


class YOLODetector:
    """
    Performs object detection using YOLO and saves bounding boxes in YOLO format.
    """
    def __init__(self, model_path):
        """
        Initializes the YOLODetector with the specified model.
        """
        self.model = YOLO(model_path)

    def detect_and_save_bboxes(self, input_folder, output_folder):
        """
        Detects objects in all images from the input folder and saves bounding boxes in YOLO format to the output folder.
        """
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Process each image in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

                # Perform detection
                results = self.model.predict(source=image_path, save=False)

                # Get original image dimensions
                image_width, image_height = results[0].orig_shape[1], results[0].orig_shape[0]

                # Write bounding box results to the output file
                with open(output_file, 'w') as f:
                    for result in results:
                        # Extract bounding boxes and sort them left-to-right by x_min
                        bboxes = sorted(
                            [(box[0].item(), box[1].item(), box[2].item(), box[3].item()) for box in result.boxes.xyxy],
                            key=lambda x: x[0]
                        )

                        # Convert bounding boxes to YOLO format and normalize
                        for bbox in bboxes:
                            x_min, y_min, x_max, y_max = bbox
                            cx = (x_min + x_max) / 2.0 / image_width  # Normalize center x
                            cy = (y_min + y_max) / 2.0 / image_height  # Normalize center y
                            w = (x_max - x_min) / image_width  # Normalize width
                            h = (y_max - y_min) / image_height  # Normalize height
                            f.write(f"0 {cx} {cy} {w} {h}\n")  # Class ID for meteors is 0

        print(f"Bounding box results saved to {output_folder}")


def main():
    """
    Main function to configure paths and run the YOLO detection.
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths for the YOLO model and input images
    models_folder = os.path.join("..", "trained_models", "yolo11m_model.pt")
    yolo_model_path = os.path.normpath(os.path.join(current_dir, models_folder))
    
    images_folder = os.path.join("..", "sample_spectrograms")
    images_path = os.path.normpath(os.path.join(current_dir, images_folder))
    
    # Configuration for input and output folders
    model_path = yolo_model_path
    input_folder = images_path
    output_folder = "sample_images_yolo_bbox_extracts"

    # Initialize the detector
    detector = YOLODetector(model_path)

    # Run detection and save bounding boxes
    detector.detect_and_save_bboxes(input_folder, output_folder)


if __name__ == "__main__":
    main()
