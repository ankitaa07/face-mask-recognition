import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_dataset(data_dir, img_size=(128, 128)):
    images, labels = [], []
    categories = ["with_mask", "without_mask"]
    
    for label, category in enumerate(categories):
        folder_path = os.path.join(data_dir, category)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} not found!")
            continue
        
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
    
    images = np.array(images, dtype="float32") / 255.0  # Normalize images
    labels = to_categorical(np.array(labels), num_classes=2)
    
    return images, labels

# Example usage
if __name__ == "__main__":
    train_data_path = "E:/dip/face-mask-recognition/dataset/train"
    test_data_path = "E:/dip/face-mask-recognition/dataset/test"
    
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        X_train, y_train = load_dataset(train_data_path)
        X_test, y_test = load_dataset(test_data_path)
        
        print(f"Training Data Loaded: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Testing Data Loaded: {X_test.shape}, Labels: {y_test.shape}")
    else:
        print("Dataset folders not found. Please check your dataset location.")
