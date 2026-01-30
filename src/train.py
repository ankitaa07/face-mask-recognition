from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from dataset_loader import load_dataset
import os

# Load data

train_data_path = r"E:\dip - Copy\face-mask-recognition\dataset\train"
test_data_path = r"E:\dip - Copy\face-mask-recognition\dataset\test"

X_train, y_train = load_dataset(train_data_path)
X_test, y_test = load_dataset(test_data_path)

# Preprocess images for MobileNetV2
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))
base_model.trainable = False  # Freeze base model

head = base_model.output
head = GlobalAveragePooling2D()(head)
head = Dense(128, activation='relu')(head)
head = Dropout(0.5)(head)
output = Dense(2, activation='softmax')(head)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=10)

model.save("mask_detector_model.h5")
