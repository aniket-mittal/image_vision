"""
CLIP Model for Attention Mask Generation

This module provides CLIP-based attention mask generation functionality.
Based directly on the working apiprompting implementation.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from typing import List, Union, Tuple, Optional

# Import the working apiprompting components
from .clip_prs.utils.factory import create_model_and_transforms, get_tokenizer
from .hook import hook_prs_logger


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


def get_model(model_name="ViT-L-14-336", layer_index=23):
    """Get the CLIP model with hooks - exactly as in original apiprompting."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pretrained = 'openai'

    # Loading Model
    model, _, preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size
    tokenizer = get_tokenizer(model_name)

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Len of res:", len(model.visual.transformer.resblocks))

    prs = hook_prs_logger(model, device, layer_index)

    return model, prs, preprocess, device, tokenizer


def gen_mask(model, prs, preprocess, device, tokenizer, image_path_or_pil_images, questions):
    """Generate masks using the exact original apiprompting approach."""
    # Load image
    images = []
    image_pils = []
    for image_path_or_pil_image in image_path_or_pil_images:
        if isinstance(image_path_or_pil_image, str):
            image_pil = Image.open(image_path_or_pil_image)
        elif isinstance(image_path_or_pil_image, Image.Image):
            image_pil = image_path_or_pil_image
        else:
            raise NotImplementedError
        image = preprocess(image_pil)[np.newaxis, :, :, :]
        images.append(image)
        image_pils.append(image_pil)
    image = torch.cat(images, dim=0).to(device)

    # Run the image:
    prs.reinit()
    with torch.no_grad():
        representation = model.encode_image(image, 
                                            attn_method='head', 
                                            normalize=False)  
                         
        attentions, mlps = prs.finalize(representation)  

    # Get the texts
    lines = questions if isinstance(questions, list) else [questions]
    print(lines[0])
    texts = tokenizer(lines).to(device)  # tokenize
    class_embeddings = model.encode_text(texts)
    class_embedding = F.normalize(class_embeddings, dim=-1)

    attention_map = attentions[:, 0, 1:, :]
    attention_map = torch.einsum('bnd,bd->bn', attention_map, class_embedding)
    HW = int(np.sqrt(attention_map.shape[1]))
    batch_size = attention_map.shape[0]
    attention_map = attention_map.view(batch_size, HW, HW)

    token_map = torch.einsum('bnd,bd->bn', mlps[:, 0, :, :], class_embedding)
    token_map = token_map.view(batch_size, HW, HW)

    return image_pils, attention_map, token_map


def merge_mask(cls_mask, patch_mask, kernel_size=3, enhance_coe=10):
    """Merge masks using original apiprompting approach."""
    cls_mask = normalize(cls_mask, "min")
    cls_mask = enhance(cls_mask, coe=enhance_coe)

    patch_mask = normalize(patch_mask, "max")

    assert kernel_size % 2 == 1
    padding_size = int((kernel_size - 1) / 2)
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding_size, padding_mode="replicate", stride=1, bias=False)
    conv.weight.data = torch.ones_like(conv.weight.data) / kernel_size**2
    conv.to(cls_mask.device)

    cls_mask = conv(cls_mask.unsqueeze(0))[0]
    patch_mask = conv(patch_mask.unsqueeze(0))[0]

    mask = normalize(cls_mask + patch_mask - cls_mask * patch_mask, "min")

    return mask


def blend_mask(image, cls_mask, patch_mask, enhance_coe, kernel_size, interpolate_method, grayscale):
    """Create the final masked image."""
    mask = merge_mask(cls_mask, patch_mask, kernel_size=kernel_size, enhance_coe=enhance_coe)
    mask = toImg(mask.detach().cpu().unsqueeze(0))
    mask = invtrans(mask, image, method=interpolate_method)
    merged_image = merge(mask.convert("L"), image.convert("RGB"), grayscale).convert("RGB")
    
    return merged_image, mask


class CLIPMaskGenerator:
    """CLIP-based attention mask generator using the working apiprompting approach."""
    
    def __init__(
        self,
        model_name: str = "ViT-L-14-336",
        layer_index: int = 22,
        device: Optional[str] = None
    ):
        """
        Initialize CLIP mask generator.
        
        Args:
            model_name: CLIP model name (e.g., "ViT-L-14-336")
            layer_index: Layer index for attention extraction
            device: Device to run on (auto-detect if None)
        """
        self.model_name = model_name
        self.layer_index = layer_index
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model using the working apiprompting approach
        self.model, self.prs, self.preprocess, self.device, self.tokenizer = get_model(
            model_name=model_name, 
            layer_index=layer_index
        )
        
        print(f"✓ Loaded CLIP model: {model_name}")
        print(f"✓ Layer index: {layer_index}")
        print(f"✓ Device: {self.device}")
    
    def generate_masks(
        self,
        images: List[Union[str, Image.Image]], 
        queries: List[str],
        enhance_coefficient: float = 5.0,
        smoothing_kernel: int = 3
    ) -> Tuple[List[Image.Image], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate attention masks using the working apiprompting approach.
        
        Args:
            images: List of image paths or PIL Images
            queries: List of text queries
            enhance_coefficient: Enhancement coefficient for mask contrast
            smoothing_kernel: Kernel size for smoothing
            
        Returns:
            Tuple of (processed_images, attention_maps, token_maps)
        """
        print(f"Generating attention masks for {len(images)} images with queries: {queries}")
        
        # Use the exact original apiprompting approach
        image_pils, attention_maps, token_maps = gen_mask(
            self.model, 
            self.prs, 
            self.preprocess, 
            self.device, 
            self.tokenizer, 
            images, 
            queries
        )
        
        print(f"✓ Generated {len(attention_maps)} attention maps")
        return image_pils, attention_maps, token_maps
    
    def create_masked_image(
        self,
        image: Image.Image,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
        grayscale_level: int = 100,
        overlay_strength: float = 1.0,
        interpolation_method: str = "LANCZOS"
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create masked image using the working apiprompting approach.
        
        Args:
            image: Original PIL image
            attention_mask: Attention mask tensor
            token_mask: Token mask tensor  
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
            token_mask, 
            enhance_coe=10.0,  # Use default enhancement
            kernel_size=3,     # Use default smoothing
            interpolate_method=interpolate_method,
            grayscale=grayscale_level
        )
        
        # Apply overlay strength if needed
        if overlay_strength < 1.0:
            image_np = np.array(image).astype(np.float32)
            masked_np = np.array(masked_image).astype(np.float32)
            blended_np = image_np * (1 - overlay_strength) + masked_np * overlay_strength
            masked_image = Image.fromarray(blended_np.astype(np.uint8))
        
        return masked_image, attention_mask_img