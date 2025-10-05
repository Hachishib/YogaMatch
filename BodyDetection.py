import cv2
import mediapipe as mp
import numpy as np
import math
import os
import glob

# Directory modified every pose
TARGET_POSE_DIR = r"C:\Users\Ayoyi\Desktop\PythonProject1\dataset\adho mukha svanasana"


# --- 1. UTILITY FUNCTION: Angle Calculation ---
def calculate_angle(a, b, c):
    """Calculates the angle (in degrees) between three 3D points."""
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Mid point (e.g., elbow)
    c = np.array(c)  # End point (e.g., wrist)

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate angle in radians
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)

    # Convert to degrees
    return np.degrees(angle)


def get_coords(landmark, index):
    """Extracts normalized (x, y) coordinates from a MediaPipe landmark list."""
    return [landmark[index].x, landmark[index].y]


# --- 2. INITIALIZATION AND TARGET POSE SETUP ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the Pose model (FIXED: Disabling smooth_segmentation to prevent crash)
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True,
    smooth_segmentation=False  # FIX for the segmentation smoothing error
)

# --- Find and Load the Target Image ---
target_pose_img = None
target_image_path = None

# Search for the first image file in the directory
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

# Create a simple mask from the target image (used for silhouette overlay)
target_pose_img_gray = cv2.cvtColor(target_pose_img, cv2.COLOR_BGR2GRAY)
_, target_mask = cv2.threshold(target_pose_img_gray, 240, 255, cv2.THRESH_BINARY_INV)

# --- 3. TARGET ANGLE EXTRACTION (For Comparison) ---
target_angles = {}
if target_results.pose_landmarks:
    target_landmarks = target_results.pose_landmarks.landmark

    # Define Landmark Indices
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

    # Extract Key Target Angles
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

    print(f"Calculated Target Angles: {target_angles}")
else:
    print("Could not detect pose landmarks in the target image. Exiting.")
    exit()

# --- 4. VIDEO CAPTURE AND MAIN LOOP ---
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    blended_img = img.copy()

    # --- Live Person Overlay (Semi-transparent red tint) ---
    if results.segmentation_mask is not None:
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        overlay_color = np.zeros(img.shape, dtype=np.uint8)
        overlay_color[:] = (255, 100, 100)

        img_float = blended_img.astype(float)
        overlay_float = overlay_color.astype(float)
        alpha = 0.3

        blended_img = np.where(condition, alpha * overlay_float + (1 - alpha) * img_float, img_float)
        blended_img = blended_img.astype(np.uint8)

    # --- Live Pose Comparison and Scoring ---
    overall_score = 0
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        score_sum = 0
        joint_count = 0
        MAX_ACCEPTABLE_DIFF = 50  # Degrees of tolerance

        # 1. Calculate Live Angles
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

                # Score calculation: 100% minus the penalty based on difference
                match_score = max(0, 100 - (angle_difference / MAX_ACCEPTABLE_DIFF) * 100)

                score_sum += match_score
                joint_count += 1

        # 2. Calculate Overall Pose Score
        overall_score = (score_sum / joint_count) if joint_count > 0 else 0

        # 3. Draw landmarks (with color feedback)
        if overall_score > 85:
            landmark_color = (0, 255, 0)  # Green
        elif overall_score > 60:
            landmark_color = (0, 165, 255)  # Orange
        else:
            landmark_color = (0, 0, 255)  # Red

        mp_drawing.draw_landmarks(
            blended_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2)
        )

    # 4. Display the Score
    score_color = (0, 255, 0) if overall_score > 85 else (0, 165, 255) if overall_score > 60 else (0, 0, 255)
    cv2.putText(blended_img, f"Downward Dog Match: {overall_score:.1f}%",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                score_color, 2, cv2.LINE_AA)

    # --- Static Silhouette Overlay (Fixes the ValueError issue) ---
    h, w, c = img.shape

    # Check if target image is valid before proceeding
    if target_pose_img.shape[0] > 0 and target_pose_img.shape[1] > 0:
        aspect_ratio = target_pose_img.shape[1] / target_pose_img.shape[0]
        new_h = int(h * 0.8)
        new_w = int(new_h * aspect_ratio)

        # Recalculate resized image and mask
        resized_static_img = cv2.resize(target_pose_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(target_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2

        alpha_static_img = 0.5

        # CRITICAL: Define the ROI to match the resized dimensions (new_h, new_w)
        # We ensure that the slicing doesn't go out of bounds of the camera frame
        y_end = min(y_offset + new_h, h)
        x_end = min(x_offset + new_w, w)

        # Adjust new_h and new_w if clipping occurred
        final_h = y_end - y_offset
        final_w = x_end - x_offset

        # Slicing the camera frame
        roi = blended_img[y_offset:y_end, x_offset:x_end]

        # Slicing the static content to match the potentially clipped ROI size
        resized_static_img_clipped = resized_static_img[:final_h, :final_w]
        resized_mask_clipped = resized_mask[:final_h, :final_w]

        # Ensure mask has 3 channels for multiplication
        mask_3_channel = cv2.cvtColor(resized_mask_clipped, cv2.COLOR_GRAY2BGR) / 255.0

        roi_float = roi.astype(np.float32)
        static_img_float = resized_static_img_clipped.astype(np.float32)

        # Check shapes one last time to prevent the crash
        if roi_float.shape == static_img_float.shape and roi_float.shape == mask_3_channel.shape:
            # Blending formula
            blended_roi_float = (static_img_float * alpha_static_img * mask_3_channel) + \
                                (roi_float * (1 - alpha_static_img * mask_3_channel))

            blended_roi_uint8 = np.clip(blended_roi_float, 0, 255).astype(np.uint8)
            blended_img[y_offset:y_end, x_offset:x_end] = blended_roi_uint8
        else:
            # This path should not be reached with the clipping logic, but useful for debugging
            print("DANGER: Shapes still mismatched. Skipping static overlay.")
            print(f"ROI: {roi_float.shape}, Static: {static_img_float.shape}, Mask: {mask_3_channel.shape}")

    # --- Display and Exit ---
    cv2.imshow("Pose Overlay and Matcher (Downward Dog)", blended_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()