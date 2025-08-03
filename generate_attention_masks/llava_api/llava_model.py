"""
LLaVA Model for Attention Mask Generation

This module provides LLaVA-based attention mask generation functionality.
Based directly on the working apiprompting implementation.
"""

import os
import re
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from typing import List, Union, Tuple, Optional
import numpy as np

# Import the working apiprompting LLaVA components
from .hook import hook_logger
from .functions import getmask, get_model


def toImg(t):
    return T.ToPILImage()(t)


def invtrans(mask, image, method=Image.BICUBIC):
    return mask.resize(image.size, method)


def merge(mask, image, grap_scale=200):
    gray = np.ones((image.size[1], image.size[0], 3)) * grap_scale
    image_np = np.array(image).astype(np.float32)[..., :3]
    mask_np = np.array(mask).astype(np.float32)
    mask_np = mask_np / 255.0
    blended_np = image_np * mask_np[:, :, None] + (1 - mask_np[:, :, None]) * gray
    blended_image = Image.fromarray((blended_np).astype(np.uint8))
    return blended_image


def normalize(mat, method="max"):
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    else:
        raise NotImplementedError


def enhance(mat, coe=10):
    mat = mat - mat.mean()
    mat = mat / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    mat = mat.clamp(0, 1)
    return mat


def revise_mask(patch_mask, kernel_size=3, enhance_coe=10):
    """Revise mask using original apiprompting approach."""
    patch_mask = normalize(patch_mask, "min")
    patch_mask = enhance(patch_mask, coe=enhance_coe)

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding_size, padding_mode="replicate", stride=1, bias=False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(patch_mask.device)

    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = patch_mask

    return mask


def blend_mask(image_path_or_pil_image, mask, key, enhance_coe, kernel_size, interpolate_method, folder):
    """Create the final masked image using original apiprompting approach."""
    mask = revise_mask(mask.float(), kernel_size=kernel_size, enhance_coe=enhance_coe)
    mask = mask.detach().cpu()
    mask = toImg(mask.reshape(1, 24, 24))

    if isinstance(image_path_or_pil_image, str):
        image = Image.open(image_path_or_pil_image)
    elif isinstance(image_path_or_pil_image, Image.Image):
        image = image_path_or_pil_image
    else:
        raise NotImplementedError

    mask = invtrans(mask, image, method=interpolate_method)
    merged_image = merge(mask.convert("L"), image.convert("RGB")).convert("RGB")

    return merged_image, mask


class LLaVAMaskGenerator:
    """LLaVA-based attention mask generator using the working apiprompting approach."""
    
    def __init__(
        self,
        model_name: str = "llava-v1.5-7b",
        layer_index: int = 20,
        device: Optional[str] = None,
        max_new_tokens: int = 20
    ):
        """
        Initialize LLaVA mask generator.
        
        Args:
            model_name: LLaVA model name (e.g., "llava-v1.5-7b")
            layer_index: Layer index for attention extraction
            device: Device to run on (auto-detect if None)
            max_new_tokens: Maximum tokens for generation
        """
        self.model_name = model_name
        self.layer_index = layer_index
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens
        
        # Load model using the working apiprompting approach
        self.tokenizer, self.model, self.image_processor, self.context_len, self.model_name = get_model(model_name)
        self.hl = hook_logger(self.model, self.model.device, layer_index=layer_index)
        
        print(f"✓ Loaded LLaVA model: {model_name}")
        print(f"✓ Layer index: {layer_index}")
        print(f"✓ Device: {self.device}")
    
    def generate_masks(
        self,
        images: List[Union[str, Image.Image]],
        queries: List[str],
        enhance_coefficient: float = 5.0,
        smoothing_kernel: int = 3
    ) -> Tuple[List[Image.Image], List[torch.Tensor], List[str]]:
        """
        Generate attention masks using the working apiprompting approach.
        
        Args:
            images: List of image paths or PIL Images
            queries: List of text queries
            enhance_coefficient: Enhancement coefficient for mask contrast
            smoothing_kernel: Kernel size for smoothing
            
        Returns:
            Tuple of (processed_images, attention_maps, model_responses)
        """
        print(f"Generating attention masks for {len(images)} images with queries: {queries}")
        
        processed_images = []
        attention_maps = []
        model_responses = []
        
        for img, query in zip(images, queries):
            # Load and preprocess image
            if isinstance(img, str):
                pil_img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                pil_img = img.convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            processed_images.append(pil_img)
            
            # Generate response and attention mask using apiprompting approach
            attention_mask, response = self._generate_single_mask(pil_img, query)
            
            attention_maps.append(attention_mask)
            model_responses.append(response)
        
        print(f"✓ Generated {len(attention_maps)} attention maps")
        return processed_images, attention_maps, model_responses
    
    def _generate_single_mask(
        self,
        image: Image.Image,
        query: str
    ) -> Tuple[torch.Tensor, str]:
        """Generate attention mask and response for a single image-query pair using apiprompting approach."""
        
        # Use the exact apiprompting approach
        with torch.no_grad():
            mask_args = type('Args', (), {
                "hl": self.hl,
                "model_name": self.model_name,
                "model": self.model,
                "tokenizer": self.tokenizer,
                "image_processor": self.image_processor,
                "context_len": self.context_len,
                "query": query,
                "conv_mode": None,
                "image_file": image,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": self.max_new_tokens,
            })()

            mask, output = getmask(mask_args)
            
        return mask, output
    
    def create_masked_image(
        self,
        image: Image.Image,
        attention_mask: torch.Tensor,
        grayscale_level: int = 100,
        overlay_strength: float = 1.0,
        interpolation_method: str = "LANCZOS"
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create masked image using the working apiprompting approach.
        
        Args:
            image: Original PIL image
            attention_mask: Attention mask tensor
            grayscale_level: Grayscale level for masking (0-255)
            overlay_strength: Strength of the overlay effect (0.0-1.0)
            interpolation_method: Interpolation method for resizing
            
        Returns:
            Tuple of (masked_image, attention_map_image)
        """
        # Use the original apiprompting blend_mask function
        interpolate_method = getattr(Image, interpolation_method, Image.LANCZOS)
        
        masked_image, attention_mask_img = blend_mask(
            image, 
            attention_mask, 
            "temp_key",  # Dummy key since we're not saving to folder
            enhance_coe=10.0,  # Use default enhancement
            kernel_size=3,     # Use default smoothing
            interpolate_method=interpolate_method,
            folder=""  # Empty folder since we're not saving
        )
        
        # Apply overlay strength if needed
        if overlay_strength < 1.0:
            image_np = np.array(image).astype(np.float32)
            masked_np = np.array(masked_image).astype(np.float32)
            blended_np = image_np * (1 - overlay_strength) + masked_np * overlay_strength
            masked_image = Image.fromarray(blended_np.astype(np.uint8))
        
        return masked_image, attention_mask_img