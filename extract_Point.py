import cv2
import mediapipe as mp
import os
import csv
import numpy as np

DATASET_DIR = r"C:\Users\johnp\OneDrive\Desktop\Projects\DA\Test Dataset\Easy\Cobra Pose"
OUTPUT_CSV = os.path.join(DATASET_DIR, "training_data.csv")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None

    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])  # Store x,y,z

    return landmarks

def process_dataset(dataset_dir):
    data = []
    class_names = []
    label_index = 0

    for root, dirs, files in os.walk(dataset_dir):
        for dirname in dirs:
            if dirname not in class_names:
                class_names.append(dirname)

        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(root, file)
                class_name = os.path.basename(os.path.dirname(path))
                label = class_names.index(class_name)

                landmarks = extract_landmarks_from_image(path)
                if landmarks:
                    data.append([label] + landmarks)
                    print(f"Processed: {path}")
                else:
                    print(f"Skipping: {path} (no pose detected)")

    return data

dataset = process_dataset(DATASET_DIR)

with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["label"] + [f"{i}" for i in range(99)])  # 33 landmarks x 3 coords
    writer.writerows(dataset)

print(f"✅ Data extraction complete. Saved to {OUTPUT_CSV}")
