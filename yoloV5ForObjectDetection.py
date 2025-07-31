import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load YOLOv5 model from torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to preprocess images for YOLOv5
def preprocess_image(image):
    # Resize image to 640x640
    image = cv2.resize(image, (640, 640))
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Function to perform object detection
def detect_objects(image_path):
    image = cv2.imread(image_path)  # Read image
    processed_image = preprocess_image(image)
    
    # Perform inference
    results = model(processed_image)
    
    # Print results
    results.print()
    
    # Save results
    results.save(save_dir='runs/detect/exp')  # Adjust save_dir as needed
    
    # Show results using OpenCV
    results_img = results.imgs[0]
    cv2.imshow('Detection Results', results_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Show results using Matplotlib
    plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Test the model on a new image
image_path = 'path/to/your/test/image.jpg'
detect_objects(image_path)