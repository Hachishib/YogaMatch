import mediapipe as mp
import cv2
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PATHS ---
TRAIN_POSITIVE = r"C:\Users\Jevon\Downloads\Dataset\TRAIN\Warrior II Pose\Positive"
TRAIN_NEGATIVE = r"C:\Users\Jevon\Downloads\Dataset\TRAIN\Warrior II Pose\Negative"
TEST_POSITIVE  = r"C:\Users\Jevon\Downloads\Dataset\TEST\Warrior II Pose\Positive"
TEST_NEGATIVE  = r"C:\Users\Jevon\Downloads\Dataset\TEST\Warrior II Pose\Negative"

MODEL_PATH = "warrior_ii_pose_detector.pkl"

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

            # Center and scale normalization
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

# --- LOAD DATA ---
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

# --- TRAIN MODEL ---
print("Training model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\n📊 CONFUSION MATRIX")
print(cm)

# Optional: Heatmap visualization
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Warrior II Pose Model")
plt.show()

# ---- MANUAL METRIC CALCULATIONS ----
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n📈 MANUAL PERFORMANCE METRICS")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Extra: Show the reliability indicators
print("\n🔍 RELIABILITY ANALYSIS")
print(f"True Positive (TP):   {TP}  → correct poses detected")
print(f"True Negative (TN):   {TN}  → incorrect poses rejected")
print(f"False Positive (FP):  {FP}  → dangerous errors (should be LOW)")
print(f"False Negative (FN):  {FN}  → strictness errors (model says wrong but user is correct)")

# --- SAVE MODEL ---
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved as {MODEL_PATH}")
