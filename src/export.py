"""
Export the DermaScope model to ONNX format.
"""
import os
import torch
import onnx
import onnxruntime as ort
import numpy as np

from .model import DermaScope
from .config import MODEL_DIR, NUM_CLASSES, IMG_SIZE, METADATA_DIM

def export_to_onnx():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'best_model_finetuned.pt')
    onnx_path = os.path.join(MODEL_DIR, 'dermascope.onnx')
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint not found at {model_path}")
        return
        
    print("Loading PyTorch model...")
    device = torch.device('cpu') # Export using CPU
    model = DermaScope(num_classes=NUM_CLASSES, metadata_dim=METADATA_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dummy inputs for tracing
    dummy_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dummy_metadata = torch.randn(1, METADATA_DIM, device=device)
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_image, dummy_metadata),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['image', 'metadata'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'metadata': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_path}")
    
    # Validate ONNX model
    print("Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
    
    # Test inference with ONNXRuntime
    print("Testing ONNXRuntime inference...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # Use numpy arrays matching dummy inputs
    ort_inputs = {
        'image': dummy_image.numpy(),
        'metadata': dummy_metadata.numpy()
    }
    
    ort_outs = ort_session.run(None, ort_inputs)
    
    # PyTorch output for comparison
    with torch.no_grad():
        pt_out = model(dummy_image, dummy_metadata).numpy()
        
    np.testing.assert_allclose(pt_out, ort_outs[0], rtol=1e-03, atol=1e-05)
    print("ONNXRuntime inference matches PyTorch output. Export successful.")

if __name__ == '__main__':
    export_to_onnx()
