#!/usr/bin/env python3
"""
Run SAM with a single click point to segment the clicked object and return a masked image
with blurred background. Prints a JSON object to stdout with fields:
{ "saved": "/abs/path.jpg", "bbox": [x1,y1,x2,y2] }
"""

import os
import sys
import json
import argparse
from typing import Tuple
import numpy as np
from PIL import Image


def apply_blur_with_mask(image: Image.Image, mask: np.ndarray, blur_strength: int = 15) -> Image.Image:
    import cv2
    image_np = np.array(image).astype(np.float32)
    h, w = image_np.shape[:2]
    if blur_strength % 2 == 0:
        blur_strength += 1
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask[:, :, 0]
    mask = np.clip(mask, 0, 1)
    if mask.shape != (h, w):
        import cv2
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(image_np, (blur_strength, blur_strength), 0)
    mask3 = np.stack([mask] * 3, axis=-1)
    out = image_np * mask3 + blurred * (1.0 - mask3)
    return Image.fromarray(out.astype(np.uint8))


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return x1, y1, x2, y2


def main(image_path: str, x: int, y: int, blur_strength: int, output_dir: str):
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except Exception as e:
        print(json.dumps({"error": f"Import SAM failed: {e}"}))
        sys.exit(1)

    # Load image
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Load SAM model
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(current_dir, "Grounded_Segment_Anything", "sam_vit_h_4b8939.pth")
        if not os.path.exists(checkpoint_path):
            # fallback to same path as grounded detector
            checkpoint_path = os.path.join(current_dir, "Grounded_Segment_Anything", "sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        predictor = SamPredictor(sam)
        predictor.set_image(image_np)
    except Exception as e:
        print(json.dumps({"error": f"Load SAM failed: {e}"}))
        sys.exit(1)

    # Predict mask from single click point
    point_coords = np.array([[x, y]])
    point_labels = np.array([1])
    try:
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
    except Exception as e:
        print(json.dumps({"error": f"Predict failed: {e}"}))
        sys.exit(1)

    mask = masks.astype(np.float32)
    if mask.ndim == 3:
        mask = mask[0]

    masked = apply_blur_with_mask(image, mask, blur_strength)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_sam_click.jpg")
    masked.save(out_path)
    x1, y1, x2, y2 = bbox_from_mask(mask)
    print(json.dumps({"saved": out_path, "bbox": [int(x1), int(y1), int(x2), int(y2)]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("--blur_strength", type=int, default=15)
    parser.add_argument("--output_dir", default="temp_output")
    args = parser.parse_args()
    main(args.image_path, args.x, args.y, args.blur_strength, args.output_dir)


