"""
Grad-CAM implementation for interpretability of DermaScope predictions.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple, List

class GradCAM:
    """
    Grad-CAM for EfficientNet backbone.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self, image: torch.Tensor, metadata: torch.Tensor, target_class: int = None) -> Tuple[np.ndarray, int]:
        """
        Generates the Grad-CAM heatmap.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image, metadata)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get pooled gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations[0].detach()
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average across channels
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        
        # ReLU to keep only positive influences
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap /= np.max(heatmap) + 1e-8
        
        return heatmap, target_class

def overlay_cam_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays the heatmap on the original image.
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB coloring
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def analyze_batch(model: torch.nn.Module, images: torch.Tensor, metadata: torch.Tensor, original_images: List[np.ndarray]):
    """
    Helper function to generate Grad-CAM for a batch of images.
    """
    # Assuming target layer is the last conv layer of EfficientNet backbone
    target_layer = list(model.backbone.features.children())[-1]
    grad_cam = GradCAM(model, target_layer)
    
    results = []
    for i in range(images.size(0)):
        img_tensor = images[i].unsqueeze(0)
        meta_tensor = metadata[i].unsqueeze(0)
        
        heatmap, pred_class = grad_cam.generate(img_tensor, meta_tensor)
        overlay = overlay_cam_on_image(original_images[i], heatmap)
        
        results.append({
            'heatmap': heatmap,
            'overlay': overlay,
            'predicted_class': pred_class
        })
        
    return results
