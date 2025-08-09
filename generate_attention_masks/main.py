#!/usr/bin/env python3
"""
Main module for attention mask generation.

This module provides the main interface for generating attention masks using CLIP models.
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(
    image: Union[str, Image.Image],
    query: str,
    layer_index: int = 22,
    enhancement_control: float = 5.0,
    smoothing_kernel: int = 3,
    grayscale_level: int = 100,
    overlay_strength: float = 1.0,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image]:
    """
    Main function for generating attention masks using CLIP.
    Args:
        image: Input image (path or PIL Image)
        query: Text query for attention generation
        layer_index: Layer index for attention extraction
        enhancement_control: Enhancement coefficient for mask contrast
        smoothing_kernel: Kernel size for smoothing
        grayscale_level: Grayscale level for masking (0-255)
        overlay_strength: Strength of the overlay effect (0.0-1.0, where 1.0 is full mask, 0.0 is original image)
        output_dir: Output directory for saving results (optional)
    Returns:
        Tuple of (attention_mask_array, blurred_image)
    """
    return run_clip(
        image=image,
        query=query,
        layer_index=layer_index,
        enhancement_control=enhancement_control,
        smoothing_kernel=smoothing_kernel,
        grayscale_level=grayscale_level,
        overlay_strength=overlay_strength
    )


def run_clip(
    image: Union[str, Image.Image],
    query: str,
    layer_index: int = 22,
    enhancement_control: float = 5.0,
    smoothing_kernel: int = 3,
    grayscale_level: int = 100,
    overlay_strength: float = 1.0
) -> Tuple[np.ndarray, Image.Image]:
    """
    Generate attention masks using CLIP model.
    
    Args:
        image: Input image (path or PIL Image)
        query: Text query for attention generation
        layer_index: Layer index for attention extraction
        enhancement_control: Enhancement coefficient for mask contrast
        smoothing_kernel: Kernel size for smoothing
        grayscale_level: Grayscale level for masking (0-255)
        overlay_strength: Strength of the overlay effect (0.0-1.0, where 1.0 is full mask, 0.0 is original image)
        
    Returns:
        Tuple of (attention_mask_array, blurred_image)
    """
    try:
        from clip_api.clip_model import CLIPMaskGenerator
        
        # Load CLIP model
        print(f"Loading CLIP model with layer_index={layer_index}...")
        generator = CLIPMaskGenerator(
            model_name="ViT-L-14-336",  # Hard-coded model name
            layer_index=layer_index,
            device="cpu"  # Use CPU for stability across environments
        )
        
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
        
        # Generate masks
        images, attention_maps, token_maps = generator.generate_masks(
            images=[pil_image],
            queries=[query],
            enhance_coefficient=enhancement_control,
            smoothing_kernel=smoothing_kernel
        )
        
        # Create masked image
        masked_img, attention_img = generator.create_masked_image(
            pil_image,
            attention_maps[0],
            token_maps[0],
            grayscale_level=grayscale_level,
            overlay_strength=overlay_strength
        )
        
        # Convert attention mask to numpy array
        attention_array = attention_maps[0].detach().cpu().numpy()
        
        print(f"✓ Generated attention mask with shape: {attention_array.shape}")
        
        return attention_array, masked_img
        
    except Exception as e:
        print(f"✗ Error in CLIP mask generation: {e}")
        import traceback
        traceback.print_exc()
        # Return empty array and original image on error
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        return np.array([]), pil_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate attention masks using CLIP models")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("query", help="Text query for attention generation")
    parser.add_argument("--layer_index", type=int, default=22,
                       help="Layer index for attention extraction")
    parser.add_argument("--enhancement_control", type=float, default=5.0,
                       help="Enhancement coefficient for mask contrast")
    parser.add_argument("--smoothing_kernel", type=int, default=3,
                       help="Kernel size for smoothing")
    parser.add_argument("--grayscale_level", type=int, default=100,
                       help="Grayscale level for masking (0-255)")
    parser.add_argument("--overlay_strength", type=float, default=1.0,
                       help="Strength of the overlay effect (0.0-1.0, where 1.0 is full mask, 0.0 is original image)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory for saving results")
    
    args = parser.parse_args()
    
    print("=== Attention Mask Generation ===")
    print(f"Image: {args.image}")
    print(f"Query: {args.query}")
    print(f"Model Type: CLIP")
    print(f"Layer Index: {args.layer_index}")
    print(f"Enhancement Control: {args.enhancement_control}")
    print(f"Smoothing Kernel: {args.smoothing_kernel}")
    print(f"Grayscale Level: {args.grayscale_level}")
    print(f"Overlay Strength: {args.overlay_strength}")
    print()
    
    try:
        attention_array, masked_image = main(
            image=args.image,
            query=args.query,
            layer_index=args.layer_index,
            enhancement_control=args.enhancement_control,
            smoothing_kernel=args.smoothing_kernel,
            grayscale_level=args.grayscale_level,
            overlay_strength=args.overlay_strength,
            output_dir=args.output_dir
        )
        
        if attention_array.size > 0:
            print("\n🎉 Successfully generated attention masks!")
            print(f"Attention mask shape: {attention_array.shape}")
            print(f"Masked image size: {masked_image.size}")
            
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(args.image))[0]
                output_path = os.path.join(args.output_dir, f"{base_name}_masked.jpg")
                masked_image.save(output_path)
                print(f"Saved masked image: {output_path}")
        else:
            print("\n❌ Failed to generate attention masks.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)