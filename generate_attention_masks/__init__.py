"""
Generate Attention Masks Library

A modern library for generating attention masks using CLIP models.
Based on the API (Attention Prompting on Image) paper for Large Vision-Language Models.
"""

from .clip_api import CLIPMaskGenerator
from .utils import ImageProcessor, MaskProcessor

__version__ = "1.0.0"
__author__ = "Generated from apiprompting library"
__description__ = "Modern attention mask generation for vision-language models"

__all__ = [
    "CLIPMaskGenerator",
    "ImageProcessor",
    "MaskProcessor"
]


def create_clip_generator(
    model_name: str = "openai/clip-vit-large-patch14-336",
    layer_index: int = 22,
    device: str = None
) -> CLIPMaskGenerator:
    """
    Create a CLIP-based mask generator.
    
    Args:
        model_name: CLIP model name
        layer_index: Layer for attention extraction
        device: Device to use
        
    Returns:
        CLIPMaskGenerator instance
    """
    return CLIPMaskGenerator(
        model_name=model_name,
        layer_index=layer_index,
        device=device
    )