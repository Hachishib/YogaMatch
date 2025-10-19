import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\johnp\Downloads\test\Training Ground\pose_model.h5")
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model as .tflite
with open("pose_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete! Saved as pose_model.tflite")