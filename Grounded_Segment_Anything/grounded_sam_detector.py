#!/usr/bin/env python3
"""
GroundingDINO + SAM Detector for object detection and segmentation.

This module provides a unified interface for using GroundingDINO for object detection
and SAM for segmentation to create precise masks for objects specified by text queries.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

# Add GroundingDINO and SAM paths
current_dir = os.path.dirname(os.path.abspath(__file__))
grounding_dino_path = os.path.join(current_dir, "GroundingDINO")
sam_path = os.path.join(current_dir, "segment_anything")

if grounding_dino_path not in sys.path:
    sys.path.append(grounding_dino_path)
if sam_path not in sys.path:
    sys.path.append(sam_path)

# Also add the groundingdino subdirectory
grounding_dino_groundingdino_path = os.path.join(grounding_dino_path, "groundingdino")
if grounding_dino_groundingdino_path not in sys.path:
    sys.path.append(grounding_dino_groundingdino_path)


class GroundedSAMDetector:
    """Unified detector for GroundingDINO + SAM."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the GroundingDINO + SAM detector.
        
        Args:
            device: Device to run on (auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._init_grounding_dino()
        self._init_sam()
        
        print(f"✓ Initialized GroundingDINO + SAM detector on {self.device}")
    
    def _init_grounding_dino(self):
        """Initialize GroundingDINO model."""
        try:
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            from groundingdino.util.inference import annotate, load_image, predict
            
            # Load GroundingDINO config and model
            config_file = os.path.join(grounding_dino_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
            checkpoint_path = os.path.join(grounding_dino_path, "groundingdino_swint_ogc.pth")
            
            if not os.path.exists(checkpoint_path):
                print("Downloading GroundingDINO model...")
                self._download_grounding_dino()
            
            args = SLConfig.fromfile(config_file)
            args.device = self.device
            
            self.grounding_dino_model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.grounding_dino_model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            self.grounding_dino_model.eval()
            self.grounding_dino_model.to(self.device)
            
            print("✓ Loaded GroundingDINO model")
            
        except Exception as e:
            print(f"✗ Error loading GroundingDINO: {e}")
            raise
    
    def _init_sam(self):
        """Initialize SAM model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Load SAM model
            sam_checkpoint_path = os.path.join(current_dir, "sam_vit_h_4b8939.pth")
            
            if not os.path.exists(sam_checkpoint_path):
                print("Downloading SAM model...")
                self._download_sam()
            
            self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            self.sam.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam)
            
            print("✓ Loaded SAM model")
            
        except Exception as e:
            print(f"✗ Error loading SAM: {e}")
            raise
    
    def _download_grounding_dino(self):
        """Download GroundingDINO model weights."""
        import urllib.request
        
        checkpoint_path = os.path.join(grounding_dino_path, "groundingdino_swint_ogc.pth")
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        print(f"Downloading GroundingDINO model from {url}...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print("✓ Downloaded GroundingDINO model")
    
    def _download_sam(self):
        """Download SAM model weights."""
        import urllib.request
        
        checkpoint_path = os.path.join(current_dir, "sam_vit_h_4b8939.pth")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        
        print(f"Downloading SAM model from {url}...")
        urllib.request.urlretrieve(url, checkpoint_path)
        print("✓ Downloaded SAM model")
    
    def detect_and_segment(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[List[np.ndarray], List[List[float]], List[str]]:
        """
        Detect objects using GroundingDINO and segment them using SAM.
        
        Args:
            image: Input image (path or PIL Image)
            text_prompt: Text prompt for object detection
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            
        Returns:
            Tuple of (masks, boxes, phrases)
        """
        # Load image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Convert PIL to numpy for GroundingDINO
        image_np = np.array(pil_image)
        
        # Detect objects with GroundingDINO
        boxes, logits, phrases = self._detect_objects(
            image_np, text_prompt, box_threshold, text_threshold
        )
        
        if len(boxes) == 0:
            return [], [], []
        
        # Segment objects with SAM
        masks = self._segment_objects(image_np, boxes)
        
        return masks, boxes, phrases
    
    def _detect_objects(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """Detect objects using GroundingDINO."""
        from groundingdino.util.inference import predict
        from groundingdino.datasets.transforms import RandomResize, ToTensor, Normalize, Compose
        from PIL import Image
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Use the same transform as load_image function
        transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Apply transform
        image_tensor, _ = transform(pil_image, None)
        
        # Run GroundingDINO prediction
        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        return boxes, logits, phrases
    
    def _segment_objects(
        self,
        image: np.ndarray,
        boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """Segment objects using SAM."""
        masks = []
        
        # Set image in SAM predictor
        self.sam_predictor.set_image(image)
        
        # Test SAM with a simple center point to see if it works
        img_h, img_w = image.shape[:2]
        center_x, center_y = img_w // 2, img_h // 2
        
        try:
            test_mask, _, _ = self.sam_predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=False
            )
        except Exception as e:
            print(f"Debug: SAM test failed: {e}")
        
        for i, box in enumerate(boxes):
            # GroundingDINO boxes are in [cx, cy, w, h] format (normalized)
            # Convert to [x1, y1, x2, y2] format for SAM
            cx, cy, w, h = box
            
            # Convert from normalized to pixel coordinates
            img_h, img_w = image.shape[:2]
            cx_px = cx * img_w
            cy_px = cy * img_h
            w_px = w * img_w
            h_px = h * img_h
            
            # Convert center format to corner format
            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2
            
            # Convert to SAM format [x, y, w, h]
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            # Generate mask using SAM
            # SAM expects box format [x1, y1, x2, y2] (corner coordinates)
            sam_box = np.array([x1, y1, x2, y2])
            
            mask, _, _ = self.sam_predictor.predict(
                box=sam_box,
                multimask_output=False
            )
            
            masks.append(mask.astype(np.float32))
        
        return masks
    
    def visualize_results(
        self,
        image: Union[str, Image.Image],
        masks: List[np.ndarray],
        boxes: List[List[float]],
        phrases: List[str]
    ) -> Image.Image:
        """
        Visualize detection and segmentation results.
        
        Args:
            image: Input image
            masks: List of segmentation masks
            boxes: List of bounding boxes
            phrases: List of detected phrases
            
        Returns:
            Visualization image
        """
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')
        
        # Convert to numpy for visualization
        image_np = np.array(pil_image)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_np)
        
        # Draw masks
        for i, mask in enumerate(masks):
            # Overlay mask
            mask_colored = np.zeros((*mask.shape, 4), dtype=np.float32)
            mask_colored[mask > 0] = [1, 0, 0, 0.3]  # Red with transparency
            ax.imshow(mask_colored)
            
            # Draw bounding box
            if i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add text label
                if i < len(phrases):
                    ax.text(x1, y1 - 10, phrases[i], 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                           fontsize=12, color='white')
        
        ax.set_title(f"GroundingDINO + SAM Results")
        ax.axis('off')
        
        # Convert matplotlib figure to PIL image
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return Image.fromarray(img_data) 