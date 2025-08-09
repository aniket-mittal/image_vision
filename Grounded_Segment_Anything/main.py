#!/usr/bin/env python3
"""
Main module for GroundingDINO + SAM segmentation mask generation.

This module provides the main interface for generating segmentation masks using 
GroundingDINO for object detection and SAM for segmentation, creating attention-like
masks that focus on specific objects in images.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
import cv2
from typing import Union, Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_oval_mask(
    mask: np.ndarray,
    padding: int = 20,
    image_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Create an oval mask around the given segmentation mask with padding.
    
    Args:
        mask: Binary segmentation mask
        padding: Padding around the mask in pixels
        image_size: Size of the original image (height, width)
        
    Returns:
        Oval mask as numpy array
    """
    if image_size is None:
        image_size = mask.shape[:2]
    
    # Ensure mask is 2D
    if len(mask.shape) > 2:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else mask[:, :, 0]
    
    # Find contours in the mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(image_size, dtype=np.float32)
    
    # Find the bounding box of all contours
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image_size[1] - x, w + 2 * padding)
    h = min(image_size[0] - y, h + 2 * padding)
    
    # Create oval mask
    oval_mask = np.zeros(image_size, dtype=np.uint8)
    
    # Calculate ellipse parameters
    center_x = x + w // 2
    center_y = y + h // 2
    axes_length = (w // 2, h // 2)
    
    # Draw ellipse
    cv2.ellipse(oval_mask, (center_x, center_y), axes_length, 0, 0, 360, 255, -1)
    
    return oval_mask.astype(np.float32) / 255.0


def apply_blur_with_mask(
    image: Image.Image,
    mask: np.ndarray,
    blur_strength: int = 15
) -> Image.Image:
    """
    Apply blur to image based on mask (similar to attention mask blurring).
    
    Args:
        image: Original PIL image
        mask: Binary mask (1 for object areas to keep clear, 0 for areas to blur)
        blur_strength: Strength of the blur effect
        
    Returns:
        Blurred PIL image
    """
    # Convert PIL image to numpy array
    image_np = np.array(image).astype(np.float32)
    h, w = image_np.shape[:2]
    
    # Ensure blur strength is odd (required by OpenCV)
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Handle mask dimensions and values
    if mask.ndim == 3:
        if mask.shape[0] == 1:  # (1, H, W) -> (H, W)
            mask = mask[0]
        else:  # (H, W, C) -> (H, W)
            mask = mask[:, :, 0]
    
    # Ensure mask is in [0, 1] range
    mask = np.clip(mask, 0, 1)
    
    # Resize mask to match image if needed
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create blurred version of the image
    image_blurred = cv2.GaussianBlur(image_np, (blur_strength, blur_strength), 0)
    
    # Expand mask to 3 channels for broadcasting
    mask3 = np.stack([mask]*3, axis=-1)
    
    # Blend: object areas (mask=1) show original image, background areas (mask=0) show blurred image
    result = image_np * mask3 + image_blurred * (1.0 - mask3)
    
    return Image.fromarray(result.astype(np.uint8))


def run_grounded_sam(
    image: Union[str, Image.Image],
    query: str,
    blur_strength: int = 15,
    padding: int = 20,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """
    Run GroundingDINO + SAM segmentation to create attention-like masks.
    
    Args:
        image: Input image (path or PIL Image)
        query: Text query for object detection
        blur_strength: Strength of the blur effect
        padding: Padding around detected objects for oval mask
        output_dir: Output directory for saving results (optional)
        
    Returns:
        Tuple of (segmentation_mask, precise_masked_image, oval_masked_image)
    """
    try:
        from grounded_sam_detector import GroundedSAMDetector
        
        # Load detector
        print(f"Loading GroundingDINO + SAM detector...")
        detector = GroundedSAMDetector()
        
        # Load image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        print(f"Processing image with query: {query}")
        
        # Detect and segment objects
        masks, boxes, phrases = detector.detect_and_segment(
            image=pil_image,
            text_prompt=query
        )
        
        if not masks or len(masks) == 0:
            print("No objects detected with the given query.")
            # Return original image with no masking
            return np.zeros(pil_image.size[::-1]), pil_image, pil_image
        
        # Combine all masks
        combined_mask = np.zeros(pil_image.size[::-1], dtype=np.float32)
        for mask in masks:
            # Remove extra dimension if present (SAM returns (1, H, W))
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            combined_mask = np.maximum(combined_mask, mask)
        
        # Create oval mask
        oval_mask = create_oval_mask(combined_mask, padding, pil_image.size[::-1])
        
        # Create precise masked image (clear mask, blurred background)
        precise_masked = apply_blur_with_mask(pil_image, combined_mask, blur_strength)
        
        # Create oval masked image (clear oval, blurred background)
        oval_masked = apply_blur_with_mask(pil_image, oval_mask, blur_strength)
        
        print(f"✓ Generated segmentation mask with shape: {combined_mask.shape}")
        
        return combined_mask, precise_masked, oval_masked
        
    except Exception as e:
        print(f"✗ Error in GroundingDINO + SAM segmentation: {e}")
        import traceback
        traceback.print_exc()
        # Return original image on error
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        return np.zeros(pil_image.size[::-1]), pil_image, pil_image


def main(
    image: Union[str, Image.Image],
    query: str,
    blur_strength: int = 15,
    padding: int = 20,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    """
    Main function for generating segmentation masks using GroundingDINO + SAM.
    
    Args:
        image: Input image (path or PIL Image)
        query: Text query for object detection
        blur_strength: Strength of the blur effect
        padding: Padding around detected objects for oval mask
        output_dir: Output directory for saving results (optional)
        
    Returns:
        Tuple of (segmentation_mask, precise_masked_image, oval_masked_image)
    """
    # Check environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != "clip_api":
        print(f"Warning: Current environment '{current_env}' may not be optimal")
        print(f"Recommended environment: clip_api")
        print("Consider running: conda activate clip_api")
    
    return run_grounded_sam(
        image=image,
        query=query,
        blur_strength=blur_strength,
        padding=padding,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO + SAM segmentation mask generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py image.jpg "dog and cat"
  
  # With custom blur strength and padding
  python main.py image.jpg "elephant" --blur_strength 20 --padding 30
        """
    )
    
    # Required arguments
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("query", help="Text query for object detection")
    
    # Optional arguments
    parser.add_argument("--blur_strength", type=int, default=15,
                       help="Strength of the blur effect (default: 15)")
    parser.add_argument("--padding", type=int, default=20,
                       help="Padding around detected objects for oval mask (default: 20)")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory for saving results (default: output)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Run segmentation
    result = main(
        image=args.image_path,
        query=args.query,
        blur_strength=args.blur_strength,
        padding=args.padding,
        output_dir=args.output_dir
    )
    
    if result[0] is None:
        sys.exit(1) 