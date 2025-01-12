#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:21:09 2024

@author: ap23710 (ashwin purushothamadhas)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, binomtest


class McNemarsTest:
    """
    Performs label comparison, McNemar's test, and visualizations for different models.
    """

    @staticmethod
    def load_labels_from_folder(folder_path):
        """
        Load all labels from a folder. Each file corresponds to an image.
        """
        labels = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    content = f.readlines()
                    labels[filename] = [line.strip() for line in content]
        return labels

    @staticmethod
    def parse_label(line):
        """
        Parse a label line in the format: 'class_id cx cy w h'.
        """
        parts = line.split()
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    @staticmethod
    def calculate_correctness_with_tolerance_and_missing(actual_labels, predicted_labels, tolerance=0.01):
        """
        Compare actual and predicted labels with a tolerance and handle missing objects.
        """
        correctness = {}
        for filename, actual_lines in actual_labels.items():
            predicted_lines = predicted_labels.get(filename, [])
            actual_parsed = [McNemarsTest.parse_label(line) for line in actual_lines]
            predicted_parsed = [McNemarsTest.parse_label(line) for line in predicted_lines]

            correctness[filename] = []
            for predicted in predicted_parsed:
                is_correct = any(
                    all(abs(predicted[i] - actual[i]) <= tolerance for i in range(1, 5))
                    for actual in actual_parsed
                )
                correctness[filename].append(1 if is_correct else 0)
        return correctness

    @staticmethod
    def perform_test(actual_labels, model1_preds, model2_preds):
        """
        Perform McNemar's Test to compare two models.
        """
        # Calculate correctness for both models
        model1_correctness = McNemarsTest.calculate_correctness_with_tolerance_and_missing(actual_labels, model1_preds)
        model2_correctness = McNemarsTest.calculate_correctness_with_tolerance_and_missing(actual_labels, model2_preds)

        # Flatten correctness lists
        model1_correct_flat = [val for sublist in model1_correctness.values() for val in sublist]
        model2_correct_flat = [val for sublist in model2_correctness.values() for val in sublist]

        # Calculate disagreements
        n_10 = sum(1 for m1, m2 in zip(model1_correct_flat, model2_correct_flat) if m1 == 1 and m2 == 0)
        n_01 = sum(1 for m1, m2 in zip(model1_correct_flat, model2_correct_flat) if m1 == 0 and m2 == 1)

        n_total = n_10 + n_01
        if n_total == 0:
            print("No disagreements to perform McNemar's test.")
            return None, None

        # McNemar's test
        z_score = (n_10 - n_01) / np.sqrt(n_total)
        p_value_binomial = binomtest(n_10, n=n_total, p=0.5, alternative='two-sided').pvalue

        print(f"\nn_10 (Model 1 correct, Model 2 incorrect): {n_10}")
        print(f"n_01 (Model 1 incorrect, Model 2 correct): {n_01}")
        print(f"z-score: {z_score:.4f}")
        print(f"Binomial Test p-value: {p_value_binomial:.4f}")

        if p_value_binomial < 0.05:
            print("There is a significant difference between the two models.")
        else:
            print("No significant difference between the two models.")

        return n_10, n_01

    @staticmethod
    def plot_test_curve(n_10, n_01, n_total, comparison_label):
        """
        Plot a binomial test curve for visualization.
        """
        x = np.arange(0, n_total + 1)
        pmf = binom.pmf(x, n=n_total, p=0.5)

        plt.figure(figsize=(12, 6))
        plt.plot(x, pmf, label="Binomial PMF", color="blue", linewidth=2)
        plt.fill_between(x, pmf, where=(x <= n_01), color="green", alpha=0.3, label="n_01 Region")
        plt.fill_between(x, pmf, where=(x >= n_10), color="red", alpha=0.3, label="n_10 Region")

        plt.axvline(n_10, color="red", linestyle="--", linewidth=2, label=f"n_10 = {n_10}")
        plt.axvline(n_01, color="green", linestyle="--", linewidth=2, label=f"n_01 = {n_01}")

        plt.title(f"Binomial Test Visualization: {comparison_label}", fontsize=16)
        plt.xlabel("Number of Correct Predictions", fontsize=14)
        plt.ylabel("Probability Mass Function (PMF)", fontsize=14)
        plt.axhline(y=0.05, color="orange", linestyle="--", label="Significance Threshold (p = 0.05)", linewidth=1.5)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=12, loc="upper right")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def count_total_objects(labels):
        """
        Count the total number of objects across all files in the labels dictionary.
        """
        return sum(len(lines) for lines in labels.values())


if __name__ == "__main__":
    """
    Main script to load labels, perform McNemar's test, and visualize results.
    """
    # Paths to folders
    actual_labels_path = "actual_labels"
    yolo_labels_path = "yolo_bbox_extracts"
    meteornet_labels_path = "meteorNet_bbox_extracts"
    fusion_labels_path = "fusion_bbox_extracts"

    # Load labels from folders
    actual_labels = McNemarsTest.load_labels_from_folder(actual_labels_path)
    yolo_preds = McNemarsTest.load_labels_from_folder(yolo_labels_path)
    meteorNet_preds = McNemarsTest.load_labels_from_folder(meteornet_labels_path)
    fusion_preds = McNemarsTest.load_labels_from_folder(fusion_labels_path)

    # Calculate total counts
    total_actual_objects = McNemarsTest.count_total_objects(actual_labels)
    total_yolo_predictions = McNemarsTest.count_total_objects(yolo_preds)
    total_meteorNet_predictions = McNemarsTest.count_total_objects(meteorNet_preds)
    total_fusion_predictions = McNemarsTest.count_total_objects(fusion_preds)

    # Display results
    print(f"Total Actual Objects: {total_actual_objects}")
    print(f"Total YOLO Predicted Coordinates: {total_yolo_predictions}")
    print(f"Total MeteorNet Predicted Coordinates: {total_meteorNet_predictions}")
    print(f"Total Fusion Predicted Coordinates: {total_fusion_predictions}")

    # Perform McNemar's Test and Plot
    print("\nMcNemar's Test: YOLO vs MeteorNet")
    n_10, n_01 = McNemarsTest.perform_test(actual_labels, yolo_preds, meteorNet_preds)
    if n_10 is not None and n_01 is not None:
        McNemarsTest.plot_test_curve(n_10, n_01, n_10 + n_01, "YOLO vs MeteorNet")

    print("\nMcNemar's Test: YOLO vs Fusion")
    n_10, n_01 = McNemarsTest.perform_test(actual_labels, yolo_preds, fusion_preds)
    if n_10 is not None and n_01 is not None:
        McNemarsTest.plot_test_curve(n_10, n_01, n_10 + n_01, "YOLO vs Fusion")

    print("\nMcNemar's Test: MeteorNet vs Fusion")
    n_10, n_01 = McNemarsTest.perform_test(actual_labels, meteorNet_preds, fusion_preds)
    if n_10 is not None and n_01 is not None:
        McNemarsTest.plot_test_curve(n_10, n_01, n_10 + n_01, "MeteorNet vs Fusion")
