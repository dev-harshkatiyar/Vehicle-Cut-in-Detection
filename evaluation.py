import torch
from pathlib import Path

# Set the paths
data_yaml = 'idd_temporal.yaml'  # Path to the dataset configuration file
weights = 'runs/train/exp/weights/best.pt'  # Path to the best weights after training

# Define the device (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, source='local')

# Function to evaluate the model
def evaluate_yolov5(model, data_yaml):
    # Run validation
    results = model.val(data=data_yaml, batch_size=16, imgsz=640, device=device)
    
    # Extract and print results
    metrics = {
        'Precision': results['metrics/precision'],
        'Recall': results['metrics/recall'],
        'mAP_0.5': results['metrics/mAP_0.5'],
        'mAP_0.5:0.95': results['metrics/mAP_0.5:0.95']
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

# Run the evaluation
if _name_ == '_main_':
    metrics = evaluate_yolov5(model, data_yaml)
    print("Evaluation completed. Metrics:", metrics)