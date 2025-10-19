import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

CSV_PATH = r"C:\Users\johnp\OneDrive\Desktop\Projects\DA\Test Dataset\Easy\Cobra Pose\training_data.csv"

# Load dataset
data = pd.read_csv(CSV_PATH)
X = data.drop('label', axis=1).values
y = data['label'].values

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Save model properly
model.save("pose_model.h5")
print("✅ Model training complete! Saved as pose_model.h5")
