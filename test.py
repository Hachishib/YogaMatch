import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load reference image
ref_img = cv2.imread(r"C:\\Users\\johnp\\OneDrive\\Desktop\\3rd Year ( School Works)\\Data Structure\\Project'\\myenv\\images.jpg")
if ref_img is None:
    print("Error: Could not load image.")
    exit()

ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
ref_results = pose.process(ref_img_rgb)

if not ref_results.pose_landmarks:
    print("No landmarks in reference image.")
    exit()

# Save reference landmarks
ref_landmarks = [(lm.x, lm.y, lm.z) for lm in ref_results.pose_landmarks.landmark]

# Function: draw stickman
def draw_stickman(img, landmarks, width, height, color=(0, 255, 0)):
    connections = mp_pose.POSE_CONNECTIONS
    landmark_points = []

    for lm in landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        landmark_points.append((x, y))

    # Draw head (circle using nose + eye distance approx)
    nose = landmark_points[0]
    left_eye = landmark_points[2]
    right_eye = landmark_points[5]
    head_radius = int(np.linalg.norm(np.array(left_eye) - np.array(right_eye)) * 2)
    cv2.circle(img, nose, head_radius, color, 2)

    # Draw body connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmark_points) and end_idx < len(landmark_points):
            cv2.line(img, landmark_points[start_idx], landmark_points[end_idx], color, 2)

# Make stickman for reference photo
ref_display = ref_img.copy()
h, w, _ = ref_img.shape
draw_stickman(ref_display, ref_results.pose_landmarks.landmark, w, h, (255, 0, 0))

# Start webcam
cap = cv2.VideoCapture(0)

def compare_landmarks(curr_landmarks, ref_landmarks, tol=0.2, match_ratio=0.7):
    """Compare only main body joints with tolerance and ratio matching"""
    # Important joint indices in Mediapipe
    key_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # shoulders, elbows, wrists, hips, knees, ankles

    matches = 0
    total = len(key_indices)

    for i in key_indices:
        curr = np.array([curr_landmarks[i].x, curr_landmarks[i].y])
        ref = np.array([ref_landmarks[i][0], ref_landmarks[i][1]])
        if np.linalg.norm(curr - ref) < tol:
            matches += 1

    # Check if enough joints match
    return (matches / total) >= match_ratio

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        draw_stickman(frame, results.pose_landmarks.landmark, w, h, (0, 255, 0))

        # Compare
        if compare_landmarks(results.pose_landmarks.landmark, ref_landmarks, tol=0.1):
            cv2.putText(frame, "✅ Pose Matched!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "❌ Not Matched", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3, cv2.LINE_AA)

    # Resize reference stickman to match webcam height
    ref_resized = cv2.resize(ref_display, (frame.shape[1] // 2, frame.shape[0]))
    frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0]))

    # Combine side by side (reference left, webcam right)
    combined = np.hstack((ref_resized, frame_resized))

    cv2.imshow("Pose Comparison", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
