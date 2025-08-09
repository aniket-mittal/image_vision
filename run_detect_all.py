#!/usr/bin/env python3
"""
Detect all relevant objects in an image using GroundingDINO + SAM and return bounding boxes.

This script loads the Grounded_Segment_Anything detector, performs detection with a generic
prompt (default: "object"), and prints a compact JSON to stdout with image size and boxes.

Boxes are returned in pixel coordinates as [x1, y1, x2, y2].
"""

import os
import sys
import json
from pathlib import Path
import argparse
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).parent
GSA_DIR = CURRENT_DIR / "Grounded_Segment_Anything"
sys.path.insert(0, str(GSA_DIR))

def _load_image_size(image_path: str):
    from PIL import Image
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        return im.size  # (width, height)

def main(image_path: str, prompt: str = "object") -> Dict[str, Any]:
    try:
        from grounded_sam_detector import GroundedSAMDetector
    except Exception as e:
        print(json.dumps({"error": f"Import error: {e}"}))
        sys.exit(1)

    width, height = _load_image_size(image_path)

    detector = GroundedSAMDetector()
    masks, boxes, phrases = detector.detect_and_segment(image=image_path, text_prompt=prompt)

    objects: List[Dict[str, Any]] = []

    # boxes are expected in normalized [cx, cy, w, h]
    for idx, box in enumerate(boxes):
        cx, cy, w_norm, h_norm = box
        x1 = max(0, int((cx - w_norm / 2) * width))
        y1 = max(0, int((cy - h_norm / 2) * height))
        x2 = min(width, int((cx + w_norm / 2) * width))
        y2 = min(height, int((cy + h_norm / 2) * height))

        objects.append({
            "id": idx,
            "bbox": [x1, y1, x2, y2],
            "label": phrases[idx] if idx < len(phrases) else "object"
        })

    return {
        "image": {
            "width": width,
            "height": height
        },
        "objects": objects
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects and return bounding boxes as JSON")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--prompt", default="object", help="Text prompt for detection (default: object)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    try:
        result = main(args.image_path, args.prompt)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


