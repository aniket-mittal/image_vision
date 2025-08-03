"""
Image processing utilities for attention mask generation.
"""

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from typing import Union, List, Tuple, Optional


class ImageProcessor:
    """Utility class for image processing operations."""
    
    def __init__(self, default_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            default_size: Default size for image resizing
        """
        self.default_size = default_size
    
    def load_image(self, image_path: Union[str, Image.Image]) -> Image.Image:
        """
        Load image from path or return PIL Image.
        
        Args:
            image_path: Path to image or PIL Image
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def preprocess_image(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image
            size: Target size (width, height)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image tensor
        """
        size = size or self.default_size
        
        transforms = [
            T.Resize(size),
            T.ToTensor()
        ]
        
        if normalize:
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        transform = T.Compose(transforms)
        return transform(image)
    
    def resize_image(
        self,
        image: Image.Image,
        size: Tuple[int, int],
        interpolation: str = "LANCZOS"
    ) -> Image.Image:
        """
        Resize image using specified interpolation.
        
        Args:
            image: PIL Image
            size: Target size (width, height)
            interpolation: Interpolation method
            
        Returns:
            Resized PIL Image
        """
        interpolation_method = getattr(Image, interpolation, Image.LANCZOS)
        return image.resize(size, interpolation_method)
    
    def save_image(
        self,
        image: Image.Image,
        save_path: str,
        quality: int = 95
    ) -> None:
        """
        Save image to specified path.
        
        Args:
            image: PIL Image to save
            save_path: Path to save image
            quality: JPEG quality (if saving as JPEG)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
            image.save(save_path, 'JPEG', quality=quality)
        else:
            image.save(save_path)
    
    def create_collage(
        self,
        images: List[Image.Image],
        titles: Optional[List[str]] = None,
        cols: int = 2
    ) -> Image.Image:
        """
        Create a collage of images.
        
        Args:
            images: List of PIL Images
            titles: Optional titles for each image
            cols: Number of columns in collage
            
        Returns:
            Collage as PIL Image
        """
        if not images:
            raise ValueError("No images provided")
        
        rows = (len(images) + cols - 1) // cols
        
        # Get max dimensions
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Create collage canvas
        collage_width = max_width * cols
        collage_height = max_height * rows
        collage = Image.new('RGB', (collage_width, collage_height), 'white')
        
        # Paste images
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            
            x = col * max_width
            y = row * max_height
            
            # Center image in cell
            paste_x = x + (max_width - img.width) // 2
            paste_y = y + (max_height - img.height) // 2
            
            collage.paste(img, (paste_x, paste_y))
        
        return collage
    
    def blend_images(
        self,
        background: Image.Image,
        overlay: Image.Image,
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Blend two images together.
        
        Args:
            background: Background image
            overlay: Overlay image
            alpha: Blending factor (0.0 to 1.0)
            
        Returns:
            Blended image
        """
        # Ensure same size
        if background.size != overlay.size:
            overlay = overlay.resize(background.size, Image.LANCZOS)
        
        # Convert to RGBA for blending
        if background.mode != 'RGBA':
            background = background.convert('RGBA')
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')
        
        # Blend
        blended = Image.blend(background, overlay, alpha)
        
        # Convert back to RGB
        return blended.convert('RGB')