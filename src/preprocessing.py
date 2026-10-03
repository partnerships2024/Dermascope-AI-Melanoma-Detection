"""
Image preprocessing techniques for skin lesions including hair removal and segmentation.
"""
import cv2
import numpy as np

def dull_razor(image: np.ndarray) -> np.ndarray:
    """
    Applies the DullRazor algorithm for hair removal using morphological black-hat filtering
    and inpainting.
    
    Args:
        image (np.ndarray): Input RGB image.
        
    Returns:
        np.ndarray: Hair-removed RGB image.
    """
    # Convert image to grayscale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    
    # Apply morphological black-hat to find hairs
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding to create mask for inpainting
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image using the mask
    inpainted_image = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    
    return inpainted_image

def segment_lesion(image: np.ndarray) -> np.ndarray:
    """
    A SAM-inspired, simplified threshold-based lesion segmentation.
    This creates a mask of the lesion.
    
    Args:
        image (np.ndarray): Input RGB image.
        
    Returns:
        np.ndarray: Binary mask of the segmented lesion.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Simple Otsu's thresholding (inverted since lesion is darker)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological opening to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return mask

def apply_background_blur(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Applies a Bokeh-like blur to the background to focus on the lesion.
    
    Args:
        image (np.ndarray): Input RGB image.
        mask (np.ndarray): Binary mask of the lesion (255 for lesion, 0 for background).
        
    Returns:
        np.ndarray: Image with blurred background.
    """
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (51, 51), 0)
    
    # Ensure mask has 3 channels for blending
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
    
    # Blend the original image and the blurred image based on the mask
    result = image * mask_3d + blurred * (1 - mask_3d)
    
    return result.astype(np.uint8)
