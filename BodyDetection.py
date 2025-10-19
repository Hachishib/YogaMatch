import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

# === Load your trained model ===
MODEL_PATH = r"C:\Users\Jevon\Downloads\Codes ni bai\Data Analytics\bound_angle_pose_detector.pkl"
model = joblib.load(MODEL_PATH)

# === Helper Functions ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def get_coords(landmark, index):
    return [landmark[index].x, landmark[index].y]

# === Mediapipe setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True,
    smooth_segmentation=False
)

# === Webcam and Timer Setup ===
cap = cv2.VideoCapture(0)
start_time = None
elapsed_time = 0.0
is_timing = False

# === Main Loop ===
while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    blended_img = img.copy()

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [coord for lm in landmarks for coord in (lm.x, lm.y)]

        prediction = model.predict([keypoints])[0]
        confidence = model.predict_proba([keypoints])[0][1] * 100  # Probability for "correct pose"

        # === Pose classification ===
        if prediction == 1 and confidence >= 90:
            feedback = f"Correct Pose ({confidence:.1f}%)"
            color = (0, 255, 0)

            # start or continue timer
            if not is_timing:
                start_time = time.time()
                is_timing = True
            else:
                elapsed_time += time.time() - start_time
                start_time = time.time()
        else:
            feedback = f"Incorrect Pose ({confidence:.1f}%)"
            color = (0, 0, 255)

            # stop timing if pose breaks
            if is_timing:
                elapsed_time += time.time() - start_time
                is_timing = False

        # === Draw body landmarks (can be disabled later) ===
       #mp_drawing.draw_landmarks(
            #blended_img,
            #results.pose_landmarks,
            #mp_pose.POSE_CONNECTIONS,
            #mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            #mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        #)

        # === Display feedback ===
        cv2.putText(blended_img, feedback, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # === Display timer ===
        total_time = elapsed_time
        if is_timing:
            total_time += time.time() - start_time

        mins = int(total_time // 60)
        secs = int(total_time % 60)
        millis = int((total_time * 1000) % 1000)
        timer_text = f"Hold Time: {mins:02}:{secs:02}.{millis:03}"

        cv2.putText(blended_img, timer_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("YogaMatch", blended_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
