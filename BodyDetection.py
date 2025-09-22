import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Drawing Utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the Pose model
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=True  # We need the segmentation mask for the semi-transparent overlay
)

# --- Load the standing image and its mask once ---
try:
    target_pose_img = cv2.imread('standing-2.jpg')
    if target_pose_img is None:
        raise FileNotFoundError("standing-2.jpg not found.")
except Exception as e:
    print(f"Error loading target image: {e}")
    exit()

# Create a simple mask from the target image (assuming a white background)
target_pose_img_gray = cv2.cvtColor(target_pose_img, cv2.COLOR_BGR2GRAY)
_, target_mask = cv2.threshold(target_pose_img_gray, 250, 255, cv2.THRESH_BINARY_INV)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    # Flip the image horizontally for a mirror effect
    img = cv2.flip(img, 1)

    # --- STEP 1: Process the raw camera frame ONLY ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    # Make a copy to add all overlays to
    processed_img = img.copy()

    # --- STEP 2: Apply the semi-transparent overlay and skeleton to the live person ---
    if results.segmentation_mask is not None:
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        overlay_color = np.zeros(img.shape, dtype=np.uint8)
        overlay_color[:] = (255, 100, 100)
        img_float = processed_img.astype(float)
        overlay_float = overlay_color.astype(float)
        alpha = 0.3

        processed_img = np.where(condition, alpha * overlay_float + (1 - alpha) * img_float, img_float)
        processed_img = processed_img.astype(np.uint8)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            processed_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # --- STEP 3: Overlay the static standing image on top of the processed frame ---
    h, w, c = img.shape

    aspect_ratio = target_pose_img.shape[1] / target_pose_img.shape[0]
    new_h = int(h * 0.8)
    new_w = int(new_h * aspect_ratio)

    resized_static_img = cv2.resize(target_pose_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(target_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2

    roi = processed_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    inv_mask = cv2.bitwise_not(resized_mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
    img_fg = cv2.bitwise_and(resized_static_img, resized_static_img, mask=resized_mask)

    dst = cv2.add(img_bg, img_fg)
    processed_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = dst

    cv2.imshow("Pose Overlay", processed_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()