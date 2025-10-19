import mediapipe as mp
import cv2
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- PATHS ---
TRAIN_POSITIVE = r"C:\Users\Jevon\Downloads\Dataset\TRAIN\Bound Angle Pose\Positive"
TRAIN_NEGATIVE = r"C:\Users\Jevon\Downloads\Dataset\TRAIN\Bound Angle Pose\Negative"
TEST_POSITIVE  = r"C:\Users\Jevon\Downloads\Dataset\TEST\Bound Angle Pose\Positive"
TEST_NEGATIVE  = r"C:\Users\Jevon\Downloads\Dataset\TEST\Bound Angle Pose\Negative"

MODEL_PATH = "bound_angle_pose_detector.pkl"

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
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y])
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
