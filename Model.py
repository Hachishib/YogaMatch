import mediapipe as mp
import cv2
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np  # ✅ Added for math operations

# --- PATHS ---
TRAIN_POSITIVE = r"C:\Users\johnp\OneDrive\Desktop\Projects\DA\Test Dataset\Inter\Garland Pose\Train Positive"
TRAIN_NEGATIVE = r"C:\Users\johnp\OneDrive\Desktop\Projects\DA\Test Dataset\Inter\Garland Pose\Train Negative"
TEST_POSITIVE  = r"C:\Users\johnp\OneDrive\Desktop\Projects\DA\Test Dataset\Inter\Garland Pose\Test Positive"
TEST_NEGATIVE  = r"C:\Users\johnp\OneDrive\Desktop\Projects\DA\Test Dataset\Inter\Garland Pose\Test Negative"

MODEL_PATH = "GarlandPose.pkl"

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_keypoints(folder, label):
    data, labels = [], []
    for img_file in os.listdir(folder):
        path = os.path.join(folder, img_file)
        image = cv2.imread(path)
        if image is None:
            continue
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Center on hips and scale by shoulder distance
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            ref_x = (left_hip.x + right_hip.x) / 2
            ref_y = (left_hip.y + right_hip.y) / 2

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            body_scale = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 +
                                 (left_shoulder.y - right_shoulder.y) ** 2)
            body_scale = max(body_scale, 1e-6)

            keypoints = []
            for lm in landmarks:
                norm_x = (lm.x - ref_x) / body_scale
                norm_y = (lm.y - ref_y) / body_scale
                keypoints.extend([norm_x, norm_y])
            

            data.append(keypoints)
            labels.append(label)
    return data, labels

print("Loading training data...")
train_data, train_labels = [], []
for folder, label in [(TRAIN_POSITIVE, 1), (TRAIN_NEGATIVE, 0)]:
    d, l = extract_keypoints(folder, label)
    train_data.extend(d)
    train_labels.extend(l)

print("Loading testing data...")
test_data, test_labels = [], []
for folder, label in [(TEST_POSITIVE, 1), (TEST_NEGATIVE, 0)]:
    d, l = extract_keypoints(folder, label)
    test_data.extend(d)
    test_labels.extend(l)

X_train = pd.DataFrame(train_data)
y_train = pd.Series(train_labels)
X_test = pd.DataFrame(test_data)
y_test = pd.Series(test_labels)

print("Training model...")
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved as {MODEL_PATH}")
