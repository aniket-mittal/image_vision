#!/usr/bin/env python3
"""
Runner script for GroundingDINO + SAM segmentation mask generation.

This script provides a simple interface to run the segmentation mask generation
from the Grounded_Segment_Anything library.
"""

import os
import sys
import argparse
from pathlib import Path

# Add Grounded_Segment_Anything to path
current_dir = Path(__file__).parent
segmentation_dir = current_dir / "Grounded_Segment_Anything"
sys.path.insert(0, str(segmentation_dir))

try:
    from Grounded_Segment_Anything.main import main
except ImportError as e:
    print(f"Error importing from Grounded_Segment_Anything: {e}")
    print("Make sure the Grounded_Segment_Anything folder is in the same directory as this script.")
    sys.exit(1)


def run_segmentation(
    image_path: str,
    query: str,
    blur_strength: int = 15,
    padding: int = 20,
    output_dir: str = "output"
):
    """
    Run GroundingDINO + SAM segmentation with the specified parameters.
    
    Args:
        image_path: Path to input image
        query: Text query for object detection
        blur_strength: Strength of the blur effect
        padding: Padding around detected objects for oval mask
        output_dir: Output directory for saving results
    """
    print("=" * 50)
    print("🎯 GROUNDINGDINO + SAM SEGMENTATION")
    print("=" * 50)
    print(f"📁 Image: {image_path}")
    print(f"💬 Query: {query}")
    print(f"🔍 Blur Strength: {blur_strength}")
    print(f"📏 Padding: {padding}")
    print(f"📤 Output Directory: {output_dir}")
    print()
    
    try:
        # Call the main function from Grounded_Segment_Anything
        segmentation_mask, precise_masked_image, oval_masked_image = main(
            image=image_path,
            query=query,
            blur_strength=blur_strength,
            padding=padding,
            output_dir=output_dir
        )
        
        if segmentation_mask.size > 0:
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            query_safe = query.replace(" ", "_").replace("/", "_").replace("\\", "_")[:20]
            param_str = f"blur{blur_strength}_pad{padding}"
            
            # Save precise masked image (clear mask, blurred background)
            precise_output_path = os.path.join(output_dir, f"{base_name}_{query_safe}_{param_str}_precise.jpg")
            precise_masked_image.save(precise_output_path)
            
            # Save oval masked image (clear oval, blurred background)
            oval_output_path = os.path.join(output_dir, f"{base_name}_{query_safe}_{param_str}_oval.jpg")
            oval_masked_image.save(oval_output_path)
            
            print("🎉 SUCCESS!")
            print(f"📊 Segmentation mask shape: {segmentation_mask.shape}")
            print(f"🖼️  Precise masked image size: {precise_masked_image.size}")
            print(f"🖼️  Oval masked image size: {oval_masked_image.size}")
            print(f"💾 Saved precise mask to: {precise_output_path}")
            print(f"💾 Saved oval mask to: {oval_output_path}")
            print()
            
            return segmentation_mask, precise_masked_image, oval_masked_image, precise_output_path, oval_output_path
        else:
            print("❌ FAILED!")
            print("No objects were detected with the given query.")
            return None, None, None, None, None
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO + SAM segmentation mask generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_segmentation.py image.jpg "dog and cat"
  
  # With custom blur strength and padding
  python run_segmentation.py image.jpg "elephant" --blur_strength 20 --padding 30
  
  # Test with the provided elephant image
  python run_segmentation.py testing_image/images/v1_116.jpg "big elephant"
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
    result = run_segmentation(
        image_path=args.image_path,
        query=args.query,
        blur_strength=args.blur_strength,
        padding=args.padding,
        output_dir=args.output_dir
    )
    
    if result[0] is None:
        sys.exit(1) 