#!/usr/bin/env python3
"""
Runner script for attention mask generation.

This script provides a simple interface to run the attention mask generation
from the generate_attention_masks library using CLIP models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add generate_attention_masks to path
current_dir = Path(__file__).parent
generate_masks_dir = current_dir / "generate_attention_masks"
sys.path.insert(0, str(generate_masks_dir))

try:
    from generate_attention_masks.main import main
except ImportError as e:
    print(f"Error importing from generate_attention_masks: {e}")
    print("Make sure the generate_attention_masks folder is in the same directory as this script.")
    sys.exit(1)


def run_attention_masks(
    image_path: str,
    query: str,
    layer_index: int = 22,
    enhancement_control: float = 5.0,
    smoothing_kernel: int = 3,
    grayscale_level: int = 100,
    overlay_strength: float = 1.0,
    output_dir: str = "output"
):
    """
    Run attention mask generation with the specified parameters.
    
    Args:
        image_path: Path to input image
        query: Text query for attention generation
        layer_index: Layer index for attention extraction
        enhancement_control: Enhancement coefficient for mask contrast
        smoothing_kernel: Kernel size for smoothing
        grayscale_level: Grayscale level for masking (0-255)
        overlay_strength: Strength of the overlay effect (0.0-1.0, where 1.0 is full mask, 0.0 is original image)
        output_dir: Output directory for saving results
    """
    print("=" * 50)
    print("🎯 ATTENTION MASK GENERATION")
    print("=" * 50)
    print(f"📁 Image: {image_path}")
    print(f"💬 Query: {query}")
    print(f"🤖 Model: CLIP")
    print(f"🔢 Layer Index: {layer_index}")
    print(f"✨ Enhancement Control: {enhancement_control}")
    print(f"🔄 Smoothing Kernel: {smoothing_kernel}")
    print(f"🎨 Grayscale Level: {grayscale_level}")
    print(f"💪 Overlay Strength: {overlay_strength}")
    print(f"📤 Output Directory: {output_dir}")
    print()
    
    try:
        # Call the main function from generate_attention_masks
        attention_array, masked_image = main(
            image=image_path,
            query=query,
            layer_index=layer_index,
            enhancement_control=enhancement_control,
            smoothing_kernel=smoothing_kernel,
            grayscale_level=grayscale_level,
            overlay_strength=overlay_strength,
            output_dir=output_dir
        )
        
        if attention_array.size > 0:
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            # Sanitize query for filename (replace spaces and special chars)
            query_safe = query.replace(" ", "_").replace("/", "_").replace("\\", "_")[:20]
            param_str = f"clip_layer{layer_index}_enh{enhancement_control}_smooth{smoothing_kernel}"
            
            output_path = os.path.join(output_dir, f"{base_name}_{query_safe}_{param_str}_masked.jpg")
            masked_image.save(output_path)
            
            print("🎉 SUCCESS!")
            print(f"📊 Attention mask shape: {attention_array.shape}")
            print(f"🖼️  Masked image size: {masked_image.size}")
            print(f"💾 Saved to: {output_path}")
            print()
            
            return attention_array, masked_image, output_path
        else:
            print("❌ FAILED!")
            print("No attention mask was generated.")
            return None, None, None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run attention mask generation using CLIP models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CLIP usage
  python run_attention_masks.py image.jpg "What is the main object?"
  
  # With custom parameters
  python run_attention_masks.py image.jpg "Describe the scene" --layer_index 20
  
  # With overlay strength control
  python run_attention_masks.py image.jpg "Find the red object" --overlay_strength 0.5
        """
    )
    
    # Required arguments
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("query", help="Text query for attention generation")
    
    # Optional arguments
    parser.add_argument("--layer_index", type=int, default=22,
                       help="Layer index for attention extraction (default: 22)")
    parser.add_argument("--enhancement_control", type=float, default=5.0,
                       help="Enhancement coefficient for mask contrast (default: 5.0)")
    parser.add_argument("--smoothing_kernel", type=int, default=3,
                       help="Kernel size for smoothing (default: 3)")

    parser.add_argument("--grayscale_level", type=int, default=100,
                       help="Grayscale level for masking 0-255 (default: 100)")
    parser.add_argument("--overlay_strength", type=float, default=1.0,
                       help="Strength of the overlay effect (0.0-1.0, where 1.0 is full mask, 0.0 is original image) (default: 1.0)")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory for saving results (default: output)")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Run attention mask generation
    result = run_attention_masks(
        image_path=args.image_path,
        query=args.query,
        layer_index=args.layer_index,
        enhancement_control=args.enhancement_control,
        smoothing_kernel=args.smoothing_kernel,
        grayscale_level=args.grayscale_level,
        overlay_strength=args.overlay_strength,
        output_dir=args.output_dir
    )
    
    if result[0] is None:
        sys.exit(1)