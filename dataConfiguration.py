import os
import torch
from pathlib import Path

# Set the paths
data_yaml = 'idd_temporal.yaml'  # Path to the dataset configuration file
weights = 'yolov5s.pt'  # Pretrained weights to start with
epochs = 50  # Number of epochs
batch_size = 16  # Batch size

# Train YOLOv5 model
def train_yolov5(data_yaml, weights, epochs, batch_size):
    # Load the YOLOv5 model from torch hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Train the model
    model.train()
    model = model.to(device)  # Move model to GPU if available

    # Setup training
    results = model.train(data=data_yaml, epochs=epochs, batch_size=batch_size, weights=weights)

    return results

# Define device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run the training
if _name_ == '_main_':
    results = train_yolov5(data_yaml, weights, epochs, batch_size)
    print("Training completed. Results:", results)