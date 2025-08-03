"""
Mask processing utilities for attention mask generation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from typing import Union, Tuple, Optional
import cv2


class MaskProcessor:
    """Utility class for mask processing operations."""
    
    def __init__(self):
        """Initialize mask processor."""
        pass
    
    def normalize_mask(
        self,
        mask: torch.Tensor,
        method: str = "min_max"
    ) -> torch.Tensor:
        """
        Normalize mask values.
        
        Args:
            mask: Input mask tensor
            method: Normalization method ("min_max", "z_score", "sigmoid")
            
        Returns:
            Normalized mask tensor
        """
        if mask.numel() == 0:
            return mask
        
        if method == "min_max":
            mask_min = mask.min()
            mask_max = mask.max()
            if mask_max > mask_min:
                return (mask - mask_min) / (mask_max - mask_min)
            else:
                return torch.zeros_like(mask)
                
        elif method == "z_score":
            mask_mean = mask.mean()
            mask_std = mask.std()
            if mask_std > 0:
                return (mask - mask_mean) / mask_std
            else:
                return mask - mask_mean
                
        elif method == "sigmoid":
            return torch.sigmoid(mask)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def enhance_contrast(
        self,
        mask: torch.Tensor,
        coefficient: float = 5.0
    ) -> torch.Tensor:
        """
        Enhance mask contrast.
        
        Args:
            mask: Input mask tensor
            coefficient: Enhancement coefficient
            
        Returns:
            Enhanced mask tensor
        """
        if mask.numel() == 0:
            return mask
        
        # Center around mean
        mask_centered = mask - mask.mean()
        
        # Scale by standard deviation
        mask_std = mask_centered.std()
        if mask_std > 0:
            mask_scaled = mask_centered / mask_std
        else:
            mask_scaled = mask_centered
        
        # Apply enhancement coefficient
        mask_enhanced = mask_scaled * coefficient
        
        # Apply sigmoid to bound values
        mask_sigmoid = torch.sigmoid(mask_enhanced)
        
        return mask_sigmoid.clamp(0, 1)
    
    def smooth_mask(
        self,
        mask: torch.Tensor,
        kernel_size: int = 3,
        method: str = "gaussian"
    ) -> torch.Tensor:
        """
        Apply smoothing to mask.
        
        Args:
            mask: Input mask tensor
            kernel_size: Size of smoothing kernel
            method: Smoothing method ("gaussian", "box", "bilateral")
            
        Returns:
            Smoothed mask tensor
        """
        if kernel_size <= 1 or mask.numel() == 0:
            return mask
        
        # Ensure mask is 2D
        original_shape = mask.shape
        if len(mask.shape) == 1:
            # Try to reshape to square
            spatial_size = int(mask.shape[0] ** 0.5)
            if spatial_size * spatial_size == mask.shape[0]:
                mask = mask.view(spatial_size, spatial_size)
            else:
                # Can't reshape - return original
                return mask
        
        if method == "gaussian":
            # Use Gaussian smoothing
            padding = kernel_size // 2
            
            # Create Gaussian kernel
            sigma = kernel_size / 3.0
            kernel_1d = torch.exp(-0.5 * ((torch.arange(kernel_size) - kernel_size // 2) / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).to(mask.device)
            
            # Apply convolution
            mask_4d = mask.unsqueeze(0).unsqueeze(0)
            smoothed = F.conv2d(mask_4d, kernel_2d, padding=padding)[0, 0]
            
        elif method == "box":
            # Use box filter
            padding = kernel_size // 2
            conv = torch.nn.Conv2d(
                1, 1,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
                bias=False
            )
            conv.weight.data = torch.ones_like(conv.weight.data) / (kernel_size ** 2)
            conv = conv.to(mask.device)
            
            smoothed = conv(mask.unsqueeze(0).unsqueeze(0))[0, 0]
            
        elif method == "bilateral":
            # Convert to numpy for bilateral filtering
            mask_np = mask.detach().cpu().numpy().astype(np.float32)
            smoothed_np = cv2.bilateralFilter(mask_np, kernel_size, 0.1, kernel_size)
            smoothed = torch.from_numpy(smoothed_np).to(mask.device)
            
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Restore original shape if needed
        if len(original_shape) == 1:
            smoothed = smoothed.flatten()
        
        return smoothed
    
    def resize_mask(
        self,
        mask: torch.Tensor,
        target_size: Tuple[int, int],
        interpolation: str = "bilinear"
    ) -> torch.Tensor:
        """
        Resize mask to target size.
        
        Args:
            mask: Input mask tensor
            target_size: Target size (height, width)
            interpolation: Interpolation method
            
        Returns:
            Resized mask tensor
        """
        if len(mask.shape) != 2:
            # Try to reshape to 2D
            if len(mask.shape) == 1:
                spatial_size = int(mask.shape[0] ** 0.5)
                if spatial_size * spatial_size == mask.shape[0]:
                    mask = mask.view(spatial_size, spatial_size)
                else:
                    raise ValueError("Cannot reshape 1D mask to 2D")
            else:
                raise ValueError(f"Mask must be 1D or 2D, got {len(mask.shape)}D")
        
        # Add batch and channel dimensions
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        
        # Resize using interpolation
        resized = F.interpolate(
            mask_4d,
            size=target_size,
            mode=interpolation,
            align_corners=False
        )
        
        return resized[0, 0]
    
    def mask_to_pil(
        self,
        mask: torch.Tensor,
        colormap: str = "jet"
    ) -> Image.Image:
        """
        Convert mask tensor to PIL Image.
        
        Args:
            mask: Input mask tensor
            colormap: Colormap to use
            
        Returns:
            PIL Image
        """
        # Normalize mask to [0, 1]
        mask_normalized = self.normalize_mask(mask, "min_max")
        
        # Convert to numpy
        mask_np = mask_normalized.detach().cpu().numpy()
        
        if colormap == "gray":
            # Grayscale
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            return Image.fromarray(mask_uint8, mode='L')
        else:
            # Apply colormap
            import matplotlib.pyplot as plt
            cm = plt.get_cmap(colormap)
            colored_mask = cm(mask_np)
            colored_mask_uint8 = (colored_mask[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored_mask_uint8, mode='RGB')
    
    def create_attention_overlay(
        self,
        image: Image.Image,
        mask: torch.Tensor,
        alpha: float = 0.4,
        colormap: str = "jet"
    ) -> Image.Image:
        """
        Create attention overlay on image.
        
        Args:
            image: Original PIL image
            mask: Attention mask tensor
            alpha: Overlay transparency
            colormap: Colormap for attention
            
        Returns:
            Image with attention overlay
        """
        # Resize mask to match image
        mask_resized = self.resize_mask(mask, (image.height, image.width))
        
        # Convert mask to colored overlay
        mask_pil = self.mask_to_pil(mask_resized, colormap)
        
        # Resize mask to match image if needed
        if mask_pil.size != image.size:
            mask_pil = mask_pil.resize(image.size, Image.LANCZOS)
        
        # Blend with original image
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        if mask_pil.mode != 'RGBA':
            mask_pil = mask_pil.convert('RGBA')
        
        blended = Image.blend(image, mask_pil, alpha)
        return blended.convert('RGB')
    
    def create_masked_image(
        self,
        image: Image.Image,
        mask: torch.Tensor,
        background_color: Union[int, Tuple[int, int, int]] = 128,
        invert_mask: bool = False
    ) -> Image.Image:
        """
        Create masked image with background replacement.
        
        Args:
            image: Original PIL image
            mask: Attention mask tensor
            background_color: Background color (grayscale int or RGB tuple)
            invert_mask: Whether to invert the mask
            
        Returns:
            Masked PIL image
        """
        # Resize mask to match image
        mask_resized = self.resize_mask(mask, (image.height, image.width))
        
        # Normalize mask
        mask_normalized = self.normalize_mask(mask_resized, "min_max")
        
        # Invert mask if requested
        if invert_mask:
            mask_normalized = 1.0 - mask_normalized
        
        # Convert to numpy
        image_np = np.array(image).astype(np.float32)
        mask_np = mask_normalized.detach().cpu().numpy().astype(np.float32)
        
        # Create background
        if isinstance(background_color, int):
            background = np.full_like(image_np, background_color)
        else:
            background = np.full_like(image_np, background_color)
        
        # Apply masking
        masked_np = image_np * mask_np[:, :, None] + background * (1 - mask_np[:, :, None])
        
        return Image.fromarray(masked_np.astype(np.uint8))