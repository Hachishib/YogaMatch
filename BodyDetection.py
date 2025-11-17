import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

# === Load your trained model ===
MODEL_PATH = r"C:\Users\johnp\Downloads\Yoga Pose\GarlandPose.pkl"
model = joblib.load(MODEL_PATH)

# === Mediapipe setup ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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
        #    Centers Hip Line 
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        ref_x = (left_hip.x + right_hip.x) / 2
        ref_y = (left_hip.y + right_hip.y) / 2
       # Sclaes Shoulder  Wides , Solve the layo
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        body_scale = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 +
                             (left_shoulder.y - right_shoulder.y) ** 2)
        body_scale = max(body_scale, 1e-6)
        #Added Z but can be remove (Para lang pag Malapit yung tao sa Camera or Hindi features)
        keypoints = []
        for lm in landmarks:
                norm_x = (lm.x - ref_x) / body_scale
                norm_y = (lm.y - ref_y) / body_scale
                keypoints.extend([norm_x, norm_y, ])
        prediction = model.predict([keypoints])[0]
        confidence = model.predict_proba([keypoints])[0][1] * 100

        # === Pose classification ===
        if prediction == 1 and confidence >= 80:
            feedback = f"Correct Pose ({confidence:.1f}%)"
            color = (0, 255, 0)
            if not is_timing:
                start_time = time.time()
                is_timing = True
            else:
                elapsed_time += time.time() - start_time
                start_time = time.time()
        else:
            feedback = f"Incorrect Pose ({confidence:.1f}%)"
            color = (0, 0, 255)
            if is_timing:
                elapsed_time += time.time() - start_time
                is_timing = False

        mp_drawing.draw_landmarks(
            blended_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        )

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
