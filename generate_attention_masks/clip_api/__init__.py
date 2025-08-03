"""
CLIP-based Attention Mask Generation API

This module provides functionality to generate attention masks using CLIP models.
"""

from .clip_model import CLIPMaskGenerator

__version__ = "1.0.0"
__all__ = ["CLIPMaskGenerator"]