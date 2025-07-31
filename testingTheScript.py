import torch
import cv2
from matplotlib import pyplot as plt

# Define the path to the trained model weights
weights = 'runs/train/exp/weights/best.pt'  # Adjust the path as necessary

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, source='local')

# Function to run inference on an image
def detect_objects(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Run inference
    results = model(img)
    
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
if _name_ == '_main_':
    test_image_path = 'path/to/your/test/image.jpg'  # Adjust the path as necessary
    detect_objects(test_image_path)