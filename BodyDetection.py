import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import time

# PALITAN NYO NALANG
TARGET_POSE_DIR = r"C:\Users\Ayoyi\Desktop\PythonProject1\dataset\adho mukha svanasana"
MAX_ACCEPTABLE_DIFF = 50
SCORE_THRESHOLD = 85


# FOR 3D detection (COSINE SIMILARITY)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# COORDINATES
def get_coords(landmark, index):
    return [landmark[index].x, landmark[index].y]


# --- 2. INITIALIZATION AND TARGET POSE SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True,
    smooth_segmentation=False
)

# ITERATION NG SA PATTERN SA FOLDER
target_pose_img = None
target_image_path = None

search_patterns = ['*.png', '*.jpg', '*.jpeg']
for pattern in search_patterns:
    files = glob.glob(os.path.join(TARGET_POSE_DIR, pattern))
    if files:
        target_image_path = files[0]
        target_pose_img = cv2.imread(target_image_path)
        break

try:
    if target_pose_img is None:
        raise FileNotFoundError(f"No image found in directory: {TARGET_POSE_DIR}. Checked for .png, .jpg, .jpeg.")

    print(f"Loaded target image: {os.path.basename(target_image_path)}")

    target_pose_img_rgb = cv2.cvtColor(target_pose_img, cv2.COLOR_BGR2RGB)
    target_results = pose.process(target_pose_img_rgb)

except Exception as e:
    print(f"Error loading target image: {e}")
    print("\nPlease verify the directory path is correct and contains valid images.")
    exit()

target_pose_img_gray = cv2.cvtColor(target_pose_img, cv2.COLOR_BGR2GRAY)
_, target_mask = cv2.threshold(target_pose_img_gray, 240, 255, cv2.THRESH_BINARY_INV)

# MANO MANONG LANDMARKS
target_angles = {}
if target_results.pose_landmarks:
    target_landmarks = target_results.pose_landmarks.landmark

    # DEFINING THE LANDMARKS
    L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    L_EL = mp_pose.PoseLandmark.LEFT_ELBOW.value
    L_WR = mp_pose.PoseLandmark.LEFT_WRIST.value
    L_HI = mp_pose.PoseLandmark.LEFT_HIP.value
    L_KN = mp_pose.PoseLandmark.LEFT_KNEE.value
    L_AN = mp_pose.PoseLandmark.LEFT_ANKLE.value

    R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    R_EL = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    R_WR = mp_pose.PoseLandmark.RIGHT_WRIST.value
    R_HI = mp_pose.PoseLandmark.RIGHT_HIP.value
    R_KN = mp_pose.PoseLandmark.RIGHT_KNEE.value
    R_AN = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    # TARGET LANDMARKS
    target_angles['left_elbow'] = calculate_angle(get_coords(target_landmarks, L_SH),
                                                  get_coords(target_landmarks, L_EL),
                                                  get_coords(target_landmarks, L_WR))
    target_angles['right_elbow'] = calculate_angle(get_coords(target_landmarks, R_SH),
                                                   get_coords(target_landmarks, R_EL),
                                                   get_coords(target_landmarks, R_WR))
    target_angles['left_knee'] = calculate_angle(get_coords(target_landmarks, L_HI), get_coords(target_landmarks, L_KN),
                                                 get_coords(target_landmarks, L_AN))
    target_angles['right_knee'] = calculate_angle(get_coords(target_landmarks, R_HI),
                                                  get_coords(target_landmarks, R_KN),
                                                  get_coords(target_landmarks, R_AN))
    target_angles['left_hip'] = calculate_angle(get_coords(target_landmarks, L_SH), get_coords(target_landmarks, L_HI),
                                                get_coords(target_landmarks, L_KN))
    target_angles['right_hip'] = calculate_angle(get_coords(target_landmarks, R_SH), get_coords(target_landmarks, R_HI),
                                                 get_coords(target_landmarks, R_KN))
    target_angles['left_shoulder'] = calculate_angle(get_coords(target_landmarks, L_EL),
                                                     get_coords(target_landmarks, L_SH),
                                                     get_coords(target_landmarks, L_HI))
    target_angles['right_shoulder'] = calculate_angle(get_coords(target_landmarks, R_EL),
                                                      get_coords(target_landmarks, R_SH),
                                                      get_coords(target_landmarks, R_HI))

else:
    print("Could not detect pose landmarks in the target image. Exiting.")
    exit()

# CAMERA SET UP AND TIMER
cap = cv2.VideoCapture(0)

# Timer variables
start_time = None
elapsed_time = 0.0
is_timing = False

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    blended_img = img.copy()

    overall_score = 0
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        score_sum = 0
        joint_count = 0

        # 1. REAL TIME CALCULATION OF ANGLES
        live_angles = {}
        live_angles['left_elbow'] = calculate_angle(get_coords(landmarks, L_SH), get_coords(landmarks, L_EL),
                                                    get_coords(landmarks, L_WR))
        live_angles['right_elbow'] = calculate_angle(get_coords(landmarks, R_SH), get_coords(landmarks, R_EL),
                                                     get_coords(landmarks, R_WR))
        live_angles['left_knee'] = calculate_angle(get_coords(landmarks, L_HI), get_coords(landmarks, L_KN),
                                                   get_coords(landmarks, L_AN))
        live_angles['right_knee'] = calculate_angle(get_coords(landmarks, R_HI), get_coords(landmarks, R_KN),
                                                    get_coords(landmarks, R_AN))
        live_angles['left_hip'] = calculate_angle(get_coords(landmarks, L_SH), get_coords(landmarks, L_HI),
                                                  get_coords(landmarks, L_KN))
        live_angles['right_hip'] = calculate_angle(get_coords(landmarks, R_SH), get_coords(landmarks, R_HI),
                                                   get_coords(landmarks, R_KN))
        live_angles['left_shoulder'] = calculate_angle(get_coords(landmarks, L_EL), get_coords(landmarks, L_SH),
                                                       get_coords(landmarks, L_HI))
        live_angles['right_shoulder'] = calculate_angle(get_coords(landmarks, R_EL), get_coords(landmarks, R_SH),
                                                        get_coords(landmarks, R_HI))

        for joint_name, target_angle in target_angles.items():
            live_angle = live_angles.get(joint_name)
            if live_angle is not None:
                angle_difference = abs(live_angle - target_angle)
                match_score = max(0, 100 - (angle_difference / MAX_ACCEPTABLE_DIFF) * 100)
                score_sum += match_score
                joint_count += 1

        overall_score = (score_sum / joint_count) if joint_count > 0 else 0

        # WALA LNG KULAY PWEDE NYO PALITAN
        if overall_score >= SCORE_THRESHOLD:
            landmark_color = (0, 255, 0)  # Green
        elif overall_score > 60:
            landmark_color = (0, 165, 255)  # Orange
        else:
            landmark_color = (0, 0, 255)  # Red

    # TIMER
    current_score = overall_score if results.pose_landmarks else 0

    if current_score >= SCORE_THRESHOLD:
        if not is_timing:
            start_time = time.time()
            is_timing = True
        else:
            elapsed_time += (time.time() - start_time)
            start_time = time.time()

        timer_status_color = (0, 255, 0)
    else:
        if is_timing:
            is_timing = False

        if current_score > 60:
            timer_status_color = (0, 165, 255)
        else:
            timer_status_color = (0, 0, 255)

    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    millis = int((elapsed_time * 1000) % 1000)

    timer_text = f"Hold Time: {mins:02}:{secs:02}.{millis:03}"

    cv2.putText(blended_img, timer_text,
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                timer_status_color, 2, cv2.LINE_AA)

    # FORM COMPARISON NA TO
    h, w, c = img.shape
    if target_pose_img.shape[0] > 0 and target_pose_img.shape[1] > 0:
        aspect_ratio = target_pose_img.shape[1] / target_pose_img.shape[0]
        new_h = int(h * 0.8)
        new_w = int(new_h * aspect_ratio)

        resized_static_img = cv2.resize(target_pose_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(target_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2

        alpha_static_img = 0.5

        y_end = min(y_offset + new_h, h)
        x_end = min(x_offset + new_w, w)

        final_h = y_end - y_offset
        final_w = x_end - x_offset

        roi = blended_img[y_offset:y_end, x_offset:x_end]

        resized_static_img_clipped = resized_static_img[:final_h, :final_w]
        resized_mask_clipped = resized_mask[:final_h, :final_w]

        mask_3_channel = cv2.cvtColor(resized_mask_clipped, cv2.COLOR_GRAY2BGR) / 255.0

        roi_float = roi.astype(np.float32)
        static_img_float = resized_static_img_clipped.astype(np.float32)

        if roi_float.shape == static_img_float.shape and roi_float.shape == mask_3_channel.shape:
            blended_roi_float = (static_img_float * alpha_static_img * mask_3_channel) + \
                                (roi_float * (1 - alpha_static_img * mask_3_channel))

            blended_roi_uint8 = np.clip(blended_roi_float, 0, 255).astype(np.uint8)
            blended_img[y_offset:y_end, x_offset:x_end] = blended_roi_uint8

    cv2.imshow("Pose Overlay and Matcher (Downward Dog)", blended_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()