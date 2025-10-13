import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import time

# palitan nyo nalang kung want nyo i try nalang
TARGET_POSE_DIR = r"C:\Users\Ayoyi\Desktop\PythonProject1\dataset\adho mukha svanasana"

# customize nyo nalang if need pa bawasan or taasan
MAX_ACCEPTABLE_DIFF = 80
SCORE_THRESHOLD = 85


# calculation ng angle
def calculate_angle(a, b, c):
    # mga landmark para makuha yung angle ng isang landmarks
    a = np.array(a)
    b = np.array(b) # this is the middle point
    c = np.array(c)

    # distance ng dalawang point like distance between a and b
    ba = a - b
    bc = c - b

    # cosine similarity na sinasabi ni sir complicated ipaliwanag search nyo nalnag kung pano gumagana
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# x and y nung landmarks
# dito magbabase kung naka gitna ba yung user or hinde
def get_coords(landmark, index):
    return [landmark[index].x, landmark[index].y]


# mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# kasama na to sa mediapipe settings kinopya ko lng
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True,
    smooth_segmentation=False
)

# mismong reference na pose initial as null
target_pose_img = None
target_image_path = None

# hinahanap yung referece  pic for comparison
search_patterns = ['*.png', '*.jpg', '*.jpeg']
for pattern in search_patterns:
    files = glob.glob(os.path.join(TARGET_POSE_DIR, pattern))
    # yung unang pic ang ginagawang references hindi lahat
    if files:
        target_image_path = files[0]
        target_pose_img = cv2.imread(target_image_path)
        break

#error handling (pwedeng tanggalin after ng training)
try:
    if target_pose_img is None:
        raise FileNotFoundError(f"No image found in directory: {TARGET_POSE_DIR}. Checked for .png, .jpg, .jpeg.")

    print(f"Loaded target image: {os.path.basename(target_image_path)}")

    target_pose_img_rgb = cv2.cvtColor(target_pose_img, cv2.COLOR_BGR2RGB)
    target_results = pose.process(target_pose_img_rgb)

#error handling (pwedeng tanggalin after ng training)
except Exception as e:
    print(f"Error loading target image: {e}")
    print("\nPlease verify the directory path is correct and contains valid images.")
    exit()

# mano manong landmarks
target_angles = {}
if target_results.pose_landmarks:
    target_landmarks = target_results.pose_landmarks.landmark

    # Defining Landmark
    # pwede dito lagay kung buong katawan ba ni user ay nakikita ng camera lagay lng if all landmark is nandon ibig sabihin buong katawan ni user nakikita ng camera
    # di lahat ng landmarks andito dagdagan nyo nalang if needed
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

    # landmarks calculation for picture
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

#error handling pwede tanggalin after training
else:
    print("Could not detect pose landmarks in the target image. Exiting.")
    exit()

# open cam
cap = cv2.VideoCapture(0)

# initial of timer variables
start_time = None
elapsed_time = 0.0
is_timing = False

# live camera frame capture
while True:
    success, img = cap.read()

    # pwede tangalin tong error handling na to
    if not success:
        print("Failed to read from camera.")
        break

    # for camera with landmarks
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    blended_img = img.copy()

    # sinasabi yung score pag nadedetect lng yung katawan pag wala edi wala
    overall_score = 0
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        score_sum = 0
        joint_count = 0

        # landmarks calculation for live (camera)
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

        # camera and picture comparison ng angles
        for joint_name, target_angle in target_angles.items():
            live_angle = live_angles.get(joint_name)
            if live_angle is not None:
                angle_difference = abs(live_angle - target_angle)
                match_score = max(0, 100 - (angle_difference / MAX_ACCEPTABLE_DIFF) * 100)
                score_sum += match_score
                joint_count += 1

        overall_score = (score_sum / joint_count) if joint_count > 0 else 0

        # wala lng kulay pwede nyo palitan sa front end
        if overall_score >= SCORE_THRESHOLD:
            landmark_color = (0, 255, 0)  # Green
        elif overall_score > 60:
            landmark_color = (0, 165, 255)  # Orange
        else:
            landmark_color = (0, 0, 255)  # Red

    # time running/stopping
    current_score = overall_score if results.pose_landmarks else 0

    if current_score >= SCORE_THRESHOLD:
        if not is_timing:
            start_time = time.time()
            is_timing = True
        else:
            elapsed_time += (time.time() - start_time)
            start_time = time.time()

        timer_status_color = (0, 255, 0)  # Green for active timing
    else:
        if is_timing:
            is_timing = False

        if current_score > 60:
            timer_status_color = (0, 165, 255)  # Orange/Yellow when paused near target
        else:
            timer_status_color = (0, 0, 255)  # Red when score is low

    # timer display
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    millis = int((elapsed_time * 1000) % 1000)

    timer_text = f"Hold Time: {mins:02}:{secs:02}.{millis:03}"

    cv2.putText(blended_img, timer_text,
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                timer_status_color, 2, cv2.LINE_AA)

    cv2.imshow("YogaMatch", blended_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()