import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


interpreter = tf.lite.Interpreter(model_path="pose_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


class_names = ["Cobra Pose"]

def landmarks_to_feature_array_strict(landmarks):
    if not landmarks or len(landmarks.landmark) != 33:
        raise ValueError("All 33 landmarks were not detected!")
    data = []
    for lm in landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data, dtype=np.float32).reshape(1, 99)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)

    try:
        features = landmarks_to_feature_array_strict(result.pose_landmarks)
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]

        label = f"{class_names[class_id]} ({confidence*100:.2f}%)"

    except ValueError as ve:
        label = str(ve)
    except Exception as e:
        label = f"Inference Error: {e}"

    # Display
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Pose Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

