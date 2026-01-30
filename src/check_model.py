from keras.models import load_model

model = load_model("mask_detector_model.h5")
model.summary()

input("\n\nPress Enter to exit...")