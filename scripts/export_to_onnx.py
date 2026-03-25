"""
Exports the trained PyTorch ASL model to ONNX format for optimized inference.
"""

import os
import sys
import json
import torch
import torch.onnx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from training.train_model import ASLClassifier

def export_to_onnx():
    model_dir = os.path.join(PROJECT_ROOT, "models", "letter_model")
    pth_path = os.path.join(model_dir, "asl_model.pth")
    config_path = os.path.join(model_dir, "config.json")
    onnx_path = os.path.join(model_dir, "asl_model.onnx")

    if not os.path.exists(pth_path) or not os.path.exists(config_path):
        print("Model or config not found. Train the model first.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loading PyTorch model...")
    device = torch.device("cpu") 
    model = ASLClassifier(
        input_size=config["input_size"],
        num_classes=config["num_classes"]
    ).to(device)
    
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, config["input_size"], device=device)

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,            \
        dummy_input,        
        onnx_path,          
        export_params=True, 
        opset_version=14,   
        do_constant_folding=True, 
        input_names = ['input'],  
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},    
                      'output' : {0 : 'batch_size'}}
    )
    print("Export successful!")

if __name__ == "__main__":
    export_to_onnx()
