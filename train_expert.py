from ultralytics import YOLO
import torch
import copy

# Configuration
model_path = '/home/sadeepa/FYP_Group_10/Amana/ultralytics/ultralytics/cfg/models/11/yolo11.yaml'
data_config = 'coco8.yaml'
save_expert1 = 'moving_expert_cv3_weights.pt'
#save_expert2 = 'expert2_weights.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_experts(model, expert1_path):
    """Extract and save cv3 weights as two experts."""
    detect_layer = model.model.model[-1]  # Get the Detect head
    expert1 = copy.deepcopy(detect_layer.cv3)  # Expert 1 (original cv3)
    #expert2 = copy.deepcopy(detect_layer.cv3)  # Expert 2 (duplicate for MoE)
    
    # Save state dicts
    torch.save(expert1.state_dict(), expert1_path)
    #torch.save(expert2.state_dict(), expert2_path)
    print(f"Experts saved to {expert1_path}")

def train_yolo():
    # Load model
    model = YOLO(model_path)
    
    # Train the model
    results = model.train(
        data=data_config,
        epochs=100,
        imgsz=640,
        device=device,
        batch=16,
        save=True,
        pretrained=True,  # Use pretrained weights if available
    )
    
    # Save cv3 weights as experts
    save_experts(model, save_expert1)
    
    return results

if __name__ == "__main__":
    train_yolo()