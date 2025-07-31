import os
import cv2
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = "IDD_Temporal"
images_path = os.path.join(dataset_path, "Images")
annotations_path = os.path.join(dataset_path, "Annotations")

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to load annotations
def load_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)
    return annotations

# Prepare dataset lists
image_files = []
annotation_files = []

for root, dirs, files in os.walk(images_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_files.append(os.path.join(root, file))

for root, dirs, files in os.walk(annotations_path):
    for file in files:
        if file.endswith(".json"):
            annotation_files.append(os.path.join(root, file))

# Ensure images and annotations match
image_files.sort()
annotation_files.sort()

# Load and preprocess data
images = []
labels = []

for image_file, annotation_file in zip(image_files, annotation_files):
    image = load_and_preprocess_image(image_file)
    annotation = load_annotations(annotation_file)
    # Assuming annotation contains bounding boxes as 'bboxes' key
    bboxes = annotation.get('bboxes', [])
    images.append(image)
    labels.append(bboxes)

images = np.array(images)

# Convert labels to numpy array (you might need to adjust this depending on your model's requirement)
labels = np.array(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save preprocessed data if needed
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")