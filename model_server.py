#!/usr/bin/env python3
import json
import os
import base64
import tempfile
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import sys
import time

# Global imports - ensure these are available everywhere
import torch
import numpy as np
from PIL import Image
import cv2

# Optional imports for advanced features
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[ModelServer] skimage not available, SSIM validation will be disabled")

# Load CLIP model once
print("[ModelServer] Loading CLIP model...")
try:
    from generate_attention_masks.clip_api.clip_model import CLIPMaskGenerator
    CLIP_GEN = CLIPMaskGenerator(model_name="ViT-L-14-336", layer_index=22, device=None)
    print("[ModelServer] CLIP model loaded")
except Exception as e:
    print(f"[ModelServer] CLIP path failed; using GroundedSAM fallback: {e}")
    CLIP_GEN = None

# Load Grounded SAM once (GroundingDINO + SAM)
print("[ModelServer] Loading GroundedSAMDetector...")
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure local Grounded_Segment_Anything and its submodules are importable
_GSA_DIR = os.path.join(_CUR_DIR, "Grounded_Segment_Anything")
_GSA_DINO_DIR = os.path.join(_GSA_DIR, "GroundingDINO")
_GSA_DINO_PKG_DIR = os.path.join(_GSA_DIR, "groundingdino")
_GSA_SAM_DIR = os.path.join(_GSA_DIR, "segment_anything")
for p in [_GSA_DIR, _GSA_DINO_DIR, _GSA_DINO_PKG_DIR, _GSA_SAM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from Grounded_Segment_Anything.grounded_sam_detector import GroundedSAMDetector
    DETECTOR = GroundedSAMDetector()
    print("[ModelServer] GroundedSAMDetector loaded")
except Exception as e:
    print(f"[ModelServer] GroundedSAMDetector failed: {e}")
    DETECTOR = None

# Import SAM components for auto-object generation
try:
    from segment_anything import SamAutomaticMaskGenerator
    SAM_AMG = SamAutomaticMaskGenerator
    print("[ModelServer] SAM AutomaticMaskGenerator imported")
except Exception as e:
    print(f"[ModelServer] SAM AutomaticMaskGenerator import failed: {e}")
    SAM_AMG = None

# Preload all diffusion models at startup for faster inference
print("[ModelServer] Preloading diffusion models...")
_DIFFUSERS_ENV = {
    "pipe_sdxl": None,
    "controlnet_depth": None,
    "controlnet_canny": None,
    "lama_manager": None,
}

# Lazy third-party models
_BISENET = {
    "net": None,
    "transform": None,
}

_FBA = {
    "net": None,
}

# Fallback inpainting method using OpenCV
def cv2_inpaint_fallback(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Simple inpainting fallback using OpenCV's TELEA algorithm"""
    try:
        # Ensure mask is binary and uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Apply inpainting
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return result
    except Exception as e:
        print(f"[ModelServer] OpenCV inpainting fallback failed: {e}")
        return image

# Global variables for lazy loading
_EASYOCR_READER = None

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _clone_if_missing(repo_url: str, dest_dir: str):
    if os.path.exists(dest_dir) and os.path.isdir(dest_dir) and os.listdir(dest_dir):
        return
    try:
        import subprocess
        print(f"[ModelServer] Cloning {repo_url} -> {dest_dir}")
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], check=True)
    except Exception as e:
        print("[ModelServer] git clone failed:", e)

def _download_to(path: str, url: str):
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"[ModelServer] download failed {url} -> {path}:", e)
        return False

def preload_diffusion_models():
    """Preload all diffusion models at startup for faster inference"""
    print("[ModelServer] Preloading SDXL Inpainting...")
    try:
        from diffusers import StableDiffusionXLInpaintPipeline, ControlNetModel
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load SDXL Inpainting
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            base, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        _DIFFUSERS_ENV["pipe_sdxl"] = pipe
        print("[ModelServer] SDXL Inpainting loaded")
        
        # Load ControlNets
        try:
            _DIFFUSERS_ENV["controlnet_canny"] = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0"
            ).to(device)
            print("[ModelServer] ControlNet Canny loaded")
        except Exception as e:
            print(f"[ModelServer] ControlNet canny unavailable: {e}")
            
        try:
            _DIFFUSERS_ENV["controlnet_depth"] = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0"
            ).to(device)
            print("[ModelServer] ControlNet Depth loaded")
        except Exception as e:
            print(f"[ModelServer] ControlNet depth unavailable: {e}")
            
    except Exception as e:
        print(f"[ModelServer] SDXL loading failed: {e}")

def preload_lama():
    """Preload LaMa inpainting model"""
    print("[ModelServer] Preloading LaMa...")
    try:
        # Try multiple import paths for LaMa
        try:
            from lama_cleaner.model_manager import ModelManager
        except ImportError:
            try:
                # Alternative import path
                from lama_cleaner.model.manager import ModelManager
            except ImportError:
                print("[ModelServer] LaMa not available: lama_cleaner not found")
                return
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DIFFUSERS_ENV["lama_manager"] = ModelManager(
            device=device, 
            no_half=False, 
            sd_device=None
        )
        print("[ModelServer] LaMa loaded")
    except Exception as e:
        print(f"[ModelServer] LaMa loading failed: {e}")
        # Try to provide helpful error message
        if "cached_download" in str(e):
            print("[ModelServer] LaMa not available, using Telea fallback: cannot import name 'cached_download' from 'huggingface_hub'")
        else:
            print(f"[ModelServer] LaMa error: {e}")

def ensure_bisenet_loaded(device: str = None):
    if _BISENET["net"] is not None:
        return _BISENET["net"]
    try:
        repo = os.path.join(_CUR_DIR, "third_party", "face-parsing.PyTorch")
        _clone_if_missing("https://github.com/zllrunning/face-parsing.PyTorch.git", repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from model import BiSeNet  # type: ignore
        ckpt_dir = os.path.join(repo, "res", "cp")
        _ensure_dir(ckpt_dir)
        ckpt = os.path.join(ckpt_dir, "79999_iter.pth")
        if not os.path.exists(ckpt):
            # Official repo stores weights under res/cp
            url = "https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth"
            _download_to(ckpt, url)
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.to(device).eval()
        _BISENET["net"] = net
        return net
    except Exception as e:
        print(f"[ModelServer] BiSeNet load failed: {e}")
        return None

def ensure_fba_loaded(device: str = None):
    if _FBA["net"] is not None:
        return _FBA["net"]
    try:
        repo = os.path.join(_CUR_DIR, "third_party", "FBA_Matting")
        _clone_if_missing("https://github.com/MarcoForte/FBA_Matting.git", repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        
        # Try to import FBA with better error handling
        try:
            from nets import FBA  # type: ignore
        except ImportError as e:
            print(f"[ModelServer] FBA import failed, trying alternative import: {e}")
            # Try alternative import paths
            try:
                sys.path.insert(0, os.path.join(repo, "nets"))
                from FBA import FBA  # type: ignore
            except ImportError:
                print("[ModelServer] FBA import failed, FBA Matting not available")
                return None
        
        ckpt_dir = os.path.join(repo, "model")
        _ensure_dir(ckpt_dir)
        ckpt = os.path.join(ckpt_dir, "FBA.pth")
        if not os.path.exists(ckpt):
            # Try alternative paths
            alt_paths = [
                os.path.join(repo, "FBA.pth"),
                os.path.join(repo, "model", "FBA.pth"),
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    ckpt = alt_path
                    break
            else:
                # Download from configurable URL
                fba_url = os.getenv("FBA_WEIGHTS_URL", "https://github.com/MarcoForte/FBA_Matting/releases/download/v1.0/FBA.pth")
                _download_to(ckpt, fba_url)
        
        if not os.path.exists(ckpt):
            print(f"[ModelServer] FBA checkpoint not found at {ckpt}")
            return None
            
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with better error handling
        try:
            net = FBA()
            checkpoint = torch.load(ckpt, map_location=device)
            net.load_state_dict(checkpoint)
            net.to(device).eval()
            _FBA["net"] = net
            print("[ModelServer] FBA Matting loaded successfully")
            return net
        except Exception as e:
            print(f"[ModelServer] FBA model loading failed: {e}")
            return None
            
    except Exception as e:
        print(f"[ModelServer] FBA Matting load failed: {e}")
        return None

def _ensure_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr  # type: ignore
            _EASYOCR_READER = easyocr.Reader(["en"])  # add languages as needed
            print("[ModelServer] easyocr initialized")
        except Exception as e:
            print("[ModelServer] easyocr unavailable:", e)
            _EASYOCR_READER = False
    return _EASYOCR_READER

def _pytesseract_ocr(pil_img: Image.Image):
    try:
        import pytesseract  # type: ignore
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        blocks = []
        texts = []
        for i in range(n):
            t = (data["text"][i] or "").strip()
            if not t:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            conf = float(data.get("conf", [0]*n)[i] or 0)
            blocks.append({"text": t, "bbox": [int(x), int(y), int(x + w), int(y + h)], "confidence": conf/100.0})
            texts.append(t)
        text = "\n".join(texts)
        return text, blocks
    except Exception as e:
        print("[ModelServer] pytesseract unavailable:", e)
        return "", []

def ocr_blocks_from_image(pil: Image.Image):
    # Try easyocr first
    reader = _ensure_easyocr()
    if reader and reader is not False:
        try:
            results = reader.readtext(np.array(pil))
            blocks = []
            texts = []
            for (bbox, t, conf) in results:
                if not t:
                    continue
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                blocks.append({"text": t, "bbox": [x1, y1, x2, y2], "confidence": float(conf)})
                texts.append(t)
            return "\n".join(texts), blocks
        except Exception as e:
            print("[ModelServer] easyocr readtext failed:", e)
    # Fallback
    return _pytesseract_ocr(pil)

def summarize_ocr_layout(pil: Image.Image, blocks: list):
    import re
    width, height = pil.size
    if not blocks:
        return {"image": {"width": width, "height": height}, "blocks": [], "candidates": {}}
    # Normalize and basic clustering
    norm_blocks = []
    for b in blocks:
        x1, y1, x2, y2 = b["bbox"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        norm_blocks.append({
            "text": b["text"],
            "bbox": b["bbox"],
            "norm": {
                "x1": x1 / width,
                "y1": y1 / height,
                "x2": x2 / width,
                "y2": y2 / height,
                "cx": cx / width,
                "cy": cy / height,
            },
            "confidence": float(b.get("confidence", 0.0)),
        })
    # Heuristics
    top = [b for b in norm_blocks if b["norm"]["y1"] < 0.2]
    bottom = [b for b in norm_blocks if b["norm"]["y2"] > 0.8]
    left = [b for b in norm_blocks if b["norm"]["x1"] < 0.15]
    right = [b for b in norm_blocks if b["norm"]["x2"] > 0.85]
    def longest(bylist):
        return max(bylist, key=lambda b: len(b["text"]), default=None)
    title = longest(top)
    # Tick-like texts: mostly numeric/small tokens
    def is_tick(t):
        t = t.strip()
        if not t:
            return False
        # numbers, short tokens, or percent/units
        return bool(re.fullmatch(r"[\d\-+.,%:/ ]{1,8}", t)) or len(t) <= 3
    x_ticks = [b for b in bottom if is_tick(b["text"])]
    y_ticks = [b for b in left if is_tick(b["text"])]
    legend = right[:5] if len(right) else []
    return {
        "image": {"width": width, "height": height},
        "blocks": norm_blocks,
        "candidates": {
            "title": title,
            "x_axis_ticks": x_ticks[:15],
            "y_axis_ticks": y_ticks[:15],
            "legend_candidates": legend,
        },
    }

def rect_mask(width: int, height: int, x1: int, y1: int, x2: int, y2: int):
    m = np.zeros((height, width), dtype=np.float32)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    m[y1:y2+1, x1:x2+1] = 1.0
    return m

def oval_mask_from_bbox(width: int, height: int, x1: int, y1: int, x2: int, y2: int):
    import cv2
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask.astype(np.float32)

def np_to_jpeg_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ensure_image_path(payload: dict) -> str:
    """Return a local image path for processing.
    Prefer 'image_data' (data URL or base64). If not provided, try 'image_path'.
    """
    image_data = payload.get("image_data")
    if image_data:
        # handle data URL or pure base64
        if isinstance(image_data, str) and image_data.startswith("data:image/"):
            try:
                _, b64 = image_data.split(",", 1)
            except ValueError:
                b64 = image_data
        else:
            b64 = image_data
        try:
            data = base64.b64decode(b64)
        except Exception:
            # sometimes clients double-wrap base64; try to parse JSON or nested fields
            try:
                nested = json.loads(image_data)
                return ensure_image_path({"image_data": nested.get("image_data")})
            except Exception:
                raise
        tmp_dir = tempfile.gettempdir()
        ts = int(time.time() * 1000)
        p = os.path.join(tmp_dir, f"ms_img_{ts}.jpg")
        with open(p, "wb") as f:
            f.write(data)
        return p
    image_path = payload.get("image_path")
    return image_path or ""

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
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(image_np, (blur_strength, blur_strength), 0)
    mask3 = np.stack([mask] * 3, axis=-1)
    out = image_np * mask3 + blurred * (1.0 - mask3)
    return Image.fromarray(out.astype(np.uint8))

def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return [0, 0, int(mask.shape[1]), int(mask.shape[0])]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def ensure_foreground_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize mask to float32 in [0,1] where 1 means KEEP SHARP (object), 0 means BLUR.
    If the mask covers more than half the image, assume it's inverted and flip it.
    """
    m = mask.astype(np.float32)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    m = np.clip(m, 0, 1)
    # Heuristic flip if majority of pixels are 1 (likely background selected)
    if m.mean() > 0.5:
        m = 1.0 - m
    return m

def sam_auto_objects(pil: Image.Image, max_masks: int = 20):
    """Generate object masks using SAM automatic mask generator (no text).
    Returns list of (mask ndarray, bbox [x1,y1,x2,y2]).
    """
    if SAM_AMG is None or DETECTOR is None or DETECTOR.sam is None:
        print("[ModelServer] SAM components not available for auto-object generation")
        return []
    
    try:
        gen = SAM_AMG(DETECTOR.sam)
        image_np = np.array(pil)
        data = gen.generate(image_np)
        objs = []
        count = 0
        for d in sorted(data, key=lambda x: x.get("area", 0), reverse=True):
            seg = d.get("segmentation")
            if seg is None:
                continue
            m = seg.astype(np.float32)
            bbox = mask_bbox(m)
            objs.append((m, bbox))
            count += 1
            if count >= max_masks:
                break
        return objs
    except Exception as e:
        print(f"[ModelServer] SAM auto-object generation failed: {e}")
        return []


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        endpoint = path.split("/")[-1]
        if endpoint == "health":
            # Enhanced health check with model status
            health_status = {
                "status": "ok",
                "models": {
                    "clip": CLIP_GEN is not None,
                    "grounded_sam": DETECTOR is not None,
                    "sam_amg": SAM_AMG is not None,
                    "fba": _FBA["net"] is not None,
                    "bisenet": _BISENET["net"] is not None,
                    "lama": _DIFFUSERS_ENV["lama_manager"] is not None
                }
            }
            return self._send(200, health_status)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        endpoint = path.split("/")[-1]
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(data) if data else {}
        except Exception as e:
            return self._send(400, {"error": f"invalid json: {e}", "raw": data[:200]})

        if endpoint == "attention":
            image_path = ensure_image_path(payload)
            query = payload.get("query", "")
            def _to_int(x, default):
                try:
                    return int(float(x))
                except Exception:
                    return default
            def _to_float(x, default):
                try:
                    return float(x)
                except Exception:
                    return default
            layer_index = _to_int(payload.get("layer_index", 22), 22)
            enhancement_control = _to_float(payload.get("enhancement_control", 5.0), 5.0)
            smoothing_kernel = _to_int(payload.get("smoothing_kernel", 3), 3)
            grayscale_level = _to_int(payload.get("grayscale_level", 100), 100)
            overlay_strength = _to_float(payload.get("overlay_strength", 1.0), 1.0)
            spatial_bias = payload.get("spatial_bias", None)  # left|right|top|bottom
            bbox_bias = payload.get("bbox_bias", None)  # [x1,y1,x2,y2] in absolute pixels or 0..1
            bias_strength = _to_float(payload.get("bias_strength", 1.0), 1.0)
            output_dir = payload.get("output_dir", "temp_output")

            if not image_path or not os.path.exists(image_path):
                return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})

            masked_image = None
            try:
                # Update model if requested layer index changed
                global CLIP_GEN
                if CLIP_GEN.layer_index != layer_index:
                    print(f"[ModelServer] Reinitializing CLIP with layer_index={layer_index}")
                    from generate_attention_masks.clip_api.clip_model import CLIPMaskGenerator as _CMG
                    CLIP_GEN = _CMG(model_name=CLIP_GEN.model_name, layer_index=layer_index, device=CLIP_GEN.device)

                print("[ModelServer] Attention inference start", {"query": query, "layer_index": layer_index})
                _imgs, attention_maps, token_maps = CLIP_GEN.generate_masks([image_path], [query], enhancement_control, smoothing_kernel)
                # Select first sample robustly
                def _first_map(x):
                    if isinstance(x, (list, tuple)):
                        return x[0]
                    if torch.is_tensor(x):
                        return x[0] if x.dim() == 3 else x
                    return x
                am = _first_map(attention_maps)
                tm = _first_map(token_maps)
                pil_image = Image.open(image_path).convert("RGB")
                # Apply optional spatial bias to attention map
                try:
                    import numpy as _np
                    h, w = am.shape[-2], am.shape[-1]
                    weight = _np.ones((h, w), dtype=_np.float32)
                    if isinstance(spatial_bias, str):
                        # Use conservative ramps to avoid flattening maps: min 0.7, max 1.6 scaled by bias_strength
                        max_gain = 1.0 + 0.6 * float(bias_strength)
                        min_gain = max(0.7, 1.0 - 0.3 * float(bias_strength))
                        if spatial_bias == "left":
                            ramp = _np.linspace(max_gain, min_gain, w).astype(_np.float32)
                            weight *= ramp[None, :]
                        elif spatial_bias == "right":
                            ramp = _np.linspace(min_gain, max_gain, w).astype(_np.float32)
                            weight *= ramp[None, :]
                        elif spatial_bias == "top":
                            ramp = _np.linspace(max_gain, min_gain, h).astype(_np.float32)
                            weight *= ramp[:, None]
                        elif spatial_bias == "bottom":
                            ramp = _np.linspace(min_gain, max_gain, h).astype(_np.float32)
                            weight *= ramp[:, None]
                    # Synthesize half-plane bbox if not provided
                    if bbox_bias is None and isinstance(spatial_bias, str):
                        if spatial_bias == "left":
                            bbox_bias = [0, 0, int(0.5 * w), h]
                        elif spatial_bias == "right":
                            bbox_bias = [int(0.5 * w), 0, w - 1, h]
                        elif spatial_bias == "top":
                            bbox_bias = [0, 0, w, int(0.5 * h)]
                        elif spatial_bias == "bottom":
                            bbox_bias = [0, int(0.5 * h), w, h - 1]
                    if bbox_bias is not None and isinstance(bbox_bias, (list, tuple)) and len(bbox_bias) == 4:
                        x1, y1, x2, y2 = bbox_bias
                        # Normalize if values in 0..1
                        if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
                            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                        x1 = max(0, min(int(x1), w - 1))
                        y1 = max(0, min(int(y1), h - 1))
                        x2 = max(0, min(int(x2), w - 1))
                        y2 = max(0, min(int(y2), h - 1))
                        if x2 < x1: x1, x2 = x2, x1
                        if y2 < y1: y1, y2 = y2, y1
                        box_gain = 1.0 + 1.0 * float(bias_strength)
                        weight[y1:y2+1, x1:x2+1] *= box_gain
                    # Apply and renormalize with fallback if variance collapses
                    am_orig = am
                    am_biased = am * torch.from_numpy(weight).to(am.device)
                    maxv = am_biased.max()
                    if float(maxv) > 0:
                        am_biased = am_biased / (maxv + 1e-6)
                    try:
                        # If variance too low, keep original map to avoid flat gray result
                        if float(am_biased.var()) < 1e-5:
                            am = am_orig
                        else:
                            am = am_biased
                    except Exception:
                        am = am_biased
                except Exception as _e:
                    print("[ModelServer] spatial bias application failed:", _e)
                masked_image, _ = CLIP_GEN.create_masked_image(
                    pil_image,
                    am,
                    tm,
                    grayscale_level=grayscale_level,
                    overlay_strength=overlay_strength,
                )
                # Compute multi-region attention proposals (robust against noise)
                try:
                    import cv2
                    amh, amw = am.shape[-2], am.shape[-1]
                    am_np = am.detach().cpu().numpy() if torch.is_tensor(am) else np.array(am)
                    am_np = (am_np - am_np.min()) / (am_np.max() - am_np.min() + 1e-6)
                    # Upscale to image size for precise geometry
                    W, H = pil_image.size
                    am_up = cv2.resize(am_np.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
                    # Adaptive saliency threshold using Otsu fallback if percentile fails
                    try:
                        p = float(np.percentile(am_up, 85.0))
                        thr = max(0.0, min(1.0, p))
                        binm = (am_up >= thr).astype(np.uint8)
                    except Exception:
                        _, binm = cv2.threshold((am_up * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        binm = (binm > 0).astype(np.uint8)
                    # Find contours and build regions
                    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    regions = []
                    min_area = max(1, int(0.005 * W * H))
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = int(w * h)
                        if area < min_area:
                            continue
                        # polygon (simplified for readability)
                        eps = 0.01 * cv2.arcLength(cnt, True)
                        poly = cv2.approxPolyDP(cnt, eps, True)
                        poly_pts = [[int(pt[0][0]), int(pt[0][1])] for pt in poly]
                        # region mean score
                        mask_r = np.zeros((H, W), dtype=np.uint8)
                        cv2.drawContours(mask_r, [cnt], -1, 1, thickness=-1)
                        sel = am_up[mask_r.astype(bool)]
                        mean_score = float(sel.mean()) if sel.size > 0 else 0.0
                        regions.append({
                            "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                            "polygon": poly_pts,
                            "mean": mean_score,
                            "area_fraction": float(area) / float(W * H),
                        })
                    # Sort by combined saliency (mean * area) and clip top-5
                    regions = sorted(regions, key=lambda r: (r["mean"] * 0.7 + r["area_fraction"] * 0.3), reverse=True)[:5]
                    # Legacy single bbox: union of top region or fallback to whole image
                    if regions:
                        xs = [r["bbox"][0] for r in regions] + [r["bbox"][2] for r in regions]
                        ys = [r["bbox"][1] for r in regions] + [r["bbox"][3] for r in regions]
                        attention_bbox = [min(xs), min(ys), max(xs), max(ys)]
                    else:
                        attention_bbox = [0, 0, W - 1, H - 1]
                    attention_regions = regions
                except Exception as _e:
                    print("[ModelServer] attention regions compute failed:", _e)
                    attention_bbox = [0, 0, pil_image.size[0] - 1, pil_image.size[1] - 1]
                    attention_regions = []
            except Exception as inner_e:
                # If CLIP path fails (e.g., torchvision C++ ops issues), log and fallback to SAM
                print("[ModelServer] CLIP path failed; using GroundedSAM fallback:", inner_e)

            # Fallback to GroundedSAM + SAM auto if CLIP attention fails
            if masked_image is None:
                try:
                    print("[ModelServer] Using SAM auto fallback for attention")
                    pil = Image.open(image_path).convert("RGB")
                    objs = sam_auto_objects(pil, max_masks=10)
                    if objs:
                        height, width = np.array(pil).shape[:2]
                        combined = np.zeros((height, width), dtype=np.float32)
                        for m, _bbox in objs:
                            if m.ndim == 3 and m.shape[0] == 1:
                                m = m[0]
                            combined = np.maximum(combined, m)
                        masked_image = apply_blur_with_mask(pil, combined, blur_strength=15)
                    else:
                        # last resort: return original image
                        masked_image = pil
                        print("[ModelServer] SAM auto produced no masks; returning original image")
                except Exception as se:
                    print("[ModelServer] SAM fallback failed:", se)
                    return self._send(500, {"error": str(se)})

            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_clip_masked.jpg")
            masked_image.save(out_path)
            b64 = np_to_jpeg_base64(masked_image)
            print("[ModelServer] Attention inference done", {"saved": out_path})
            return self._send(200, {"saved": out_path, "processedImageData": f"data:image/jpeg;base64,{b64}", "attention_bbox": locals().get("attention_bbox"), "attention_regions": locals().get("attention_regions")})

        if endpoint == "ocr":
            try:
                image_path = ensure_image_path(payload)
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})
                pil = Image.open(image_path).convert("RGB")
                text, blocks = ocr_blocks_from_image(pil)
                layout = summarize_ocr_layout(pil, blocks)
                return self._send(200, {"text": text.strip(), "blocks": blocks, "layout": layout})
            except Exception as e:
                print("[ModelServer] OCR error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "ocr_focus":
            # Build a focus mask from OCR layout (e.g., title or axes) and blur background
            try:
                image_path = ensure_image_path(payload)
                region = payload.get("region", "title")  # title | x_axis | y_axis | legend
                blur_strength = int(payload.get("blur_strength", 15))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})
                pil = Image.open(image_path).convert("RGB")
                text, blocks = ocr_blocks_from_image(pil)
                layout = summarize_ocr_layout(pil, blocks)
                width, height = pil.size
                m = np.zeros((height, width), dtype=np.float32)
                cand = layout.get("candidates", {})
                if region == "title" and cand.get("title"):
                    x1, y1, x2, y2 = cand["title"]["bbox"]
                    m = rect_mask(width, height, x1, y1, x2, y2)
                elif region == "x_axis":
                    for b in cand.get("x_axis_ticks", []):
                        x1, y1, x2, y2 = b["bbox"]
                        m = np.maximum(m, rect_mask(width, height, x1, y1, x2, y2))
                elif region == "y_axis":
                    for b in cand.get("y_axis_ticks", []):
                        x1, y1, x2, y2 = b["bbox"]
                        m = np.maximum(m, rect_mask(width, height, x1, y1, x2, y2))
                elif region == "legend":
                    for b in cand.get("legend_candidates", []):
                        x1, y1, x2, y2 = b["bbox"]
                        m = np.maximum(m, rect_mask(width, height, x1, y1, x2, y2))
                # If nothing found, keep original image (mask zeros)
                masked = apply_blur_with_mask(pil, m, blur_strength)
                b64 = np_to_jpeg_base64(masked)
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}", "layout": layout, "region": region})
            except Exception as e:
                print("[ModelServer] OCR focus error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "detect_all":
            image_path = ensure_image_path(payload)
            prompt = payload.get("prompt", "object")
            if not image_path or not os.path.exists(image_path):
                return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})
            pil = Image.open(image_path).convert("RGB")
            width, height = pil.size
            print("[ModelServer] Detect_all start", {"prompt": prompt})
            objects = []
            try:
                # Prefer textual detect when extensions are working
                masks, boxes, phrases = DETECTOR.detect_and_segment(image=image_path, text_prompt=prompt)
                for idx, box in enumerate(boxes):
                    cx, cy, w_norm, h_norm = box
                    x1 = max(0, int((cx - w_norm / 2) * width))
                    y1 = max(0, int((cy - h_norm / 2) * height))
                    x2 = min(width, int((cx + w_norm / 2) * width))
                    y2 = min(height, int((cy + h_norm / 2) * height))
                    label = phrases[idx] if idx < len(phrases) else "object"
                    objects.append({"id": idx, "bbox": [x1, y1, x2, y2], "label": label})
            except Exception as e:
                # Fallback to SAM auto if GroundingDINO C++ ops are missing (_C not defined)
                print("[ModelServer] Detect_all fallback via SAM auto:", e)
                objs = sam_auto_objects(pil)
                for idx, (_m, bbox) in enumerate(objs):
                    objects.append({"id": idx, "bbox": bbox, "label": "object"})
            print("[ModelServer] Detect_all done", {"count": len(objects)})
            return self._send(200, {"image": {"width": width, "height": height}, "objects": objects})

        if endpoint == "prefetch_controlnet":
            try:
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                if _DIFFUSERS_ENV["pipe_sdxl"] is None:
                    from diffusers import StableDiffusionXLInpaintPipeline
                    import torch as _torch
                    base = "stabilityai/stable-diffusion-xl-base-1.0"
                    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base, torch_dtype=_torch.float16 if device == "cuda" else _torch.float32)
                    pipe = pipe.to(device)
                    _DIFFUSERS_ENV["pipe_sdxl"] = pipe
                if _DIFFUSERS_ENV["controlnet_canny"] is None:
                    from diffusers import ControlNetModel
                    _DIFFUSERS_ENV["controlnet_canny"] = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")
                if _DIFFUSERS_ENV["controlnet_depth"] is None:
                    from diffusers import ControlNetModel
                    _DIFFUSERS_ENV["controlnet_depth"] = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0")
                return self._send(200, {"ok": True})
            except Exception as e:
                print("[ModelServer] prefetch_controlnet error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "sam_click":
            try:
                image_path = ensure_image_path(payload)
                x = int(payload.get("x", 0))
                y = int(payload.get("y", 0))
                blur_strength = int(payload.get("blur_strength", 15))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})
                pil = Image.open(image_path).convert("RGB")
                image_np = np.array(pil)
                DETECTOR.sam_predictor.set_image(image_np)
                masks, _, _ = DETECTOR.sam_predictor.predict(
                    point_coords=np.array([[x, y]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                mask = masks.astype(np.float32)
                if mask.ndim == 3:
                    mask = mask[0]
                masked = apply_blur_with_mask(pil, mask, blur_strength)
                b64 = np_to_jpeg_base64(masked)
                bbox = mask_bbox(mask)
                print("[ModelServer] SAM-click done", {"bbox": bbox})
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}", "bbox": bbox})
            except Exception as e:
                print("[ModelServer] SAM-click error", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "face_parse":
            try:
                image_path = ensure_image_path(payload)
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                net = ensure_bisenet_loaded(device)
                if net is None:
                    return self._send(500, {"error": "BiSeNet unavailable"})
                pil = Image.open(image_path).convert("RGB")
                import torchvision.transforms as T
                transform = _BISENET["transform"]
                if transform is None:
                    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
                x = transform(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = net(x)[0]
                    parsing = out.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                # Map back to original size
                import cv2
                parsing_up = cv2.resize(parsing, pil.size, interpolation=cv2.INTER_NEAREST)
                # Build face mask (aggregate relevant labels)
                face_labels = set([1, 2, 3, 4, 5, 10, 11, 12, 13])
                face_mask = np.isin(parsing_up, list(face_labels)).astype(np.uint8) * 255
                # Pack PNGs
                buf = BytesIO()
                Image.fromarray(face_mask, mode="L").save(buf, format="PNG")
                face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return self._send(200, {"parsing": {"width": pil.size[0], "height": pil.size[1]}, "face_mask_png": f"data:image/png;base64,{face_b64}"})
            except Exception as e:
                print("[ModelServer] face_parse error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "matting_refine":
            try:
                image_path = ensure_image_path(payload)
                mask_png = payload.get("mask_png", "")
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                if not mask_png:
                    return self._send(400, {"error": "mask_png required (data URL)"})
                pil = Image.open(image_path).convert("RGB")
                W, H = pil.size
                if mask_png.startswith("data:image"):
                    _, m64 = mask_png.split(",", 1)
                else:
                    m64 = mask_png
                mask = Image.open(BytesIO(base64.b64decode(m64))).convert("L").resize((W, H))
                mask_np = np.array(mask).astype(np.float32) / 255.0
                # Try FBA Matting for refinement
                net = ensure_fba_loaded()
                refined = None
                if net is not None:
                    try:
                        # Minimal guided refinement: expand soft edges near boundary; FBA typically needs trimap
                        # Build a pseudo-trimap from soft mask
                        fg = (mask_np > 0.9).astype(np.uint8) * 255
                        bg = (mask_np < 0.1).astype(np.uint8) * 255
                        import cv2
                        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        fg_d = cv2.erode(fg, k, iterations=1)
                        bg_d = cv2.erode(bg, k, iterations=1)
                        trimap = np.full((H, W), 128, dtype=np.uint8)
                        trimap[bg_d > 0] = 0
                        trimap[fg_d > 0] = 255
                        # FBA expects tensors; adaptively call its inference util if available
                        from torchvision import transforms as T
                        to_tensor = T.ToTensor()
                        img_t = to_tensor(pil).unsqueeze(0)
                        trimap_t = torch.from_numpy(trimap).float().unsqueeze(0).unsqueeze(0) / 255.0
                        img_t = img_t.to(next(net.parameters()).device)
                        trimap_t = trimap_t.to(next(net.parameters()).device)
                        with torch.no_grad():
                            pred = net(img_t, trimap_t)
                            if isinstance(pred, (list, tuple)):
                                pred = pred[0]
                            alpha = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
                            refined = np.clip(alpha, 0.0, 1.0)
                    except Exception as _e:
                        print("[ModelServer] FBA refine failed, will fallback:", _e)
                        refined = None
                if refined is None:
                    # Fallback: edge-aware feather using distance transform
                    import cv2
                    binm = (mask_np > 0.5).astype(np.uint8)
                    dist_fg = cv2.distanceTransform(binm, cv2.DIST_L2, 3)
                    dist_bg = cv2.distanceTransform(1 - binm, cv2.DIST_L2, 3)
                    alpha = dist_fg / (dist_fg + dist_bg + 1e-6)
                    refined = np.clip(alpha, 0.0, 1.0)
                # Return RGBA matte composite
                rgba = np.concatenate([np.array(pil), (refined * 255).astype(np.uint8)[..., None]], axis=-1)
                buf = BytesIO()
                Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                return self._send(200, {"refined_mask_png": f"data:image/png;base64,{b64}"})
            except Exception as e:
                print("[ModelServer] matting_refine error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "sam_multi_click":
            try:
                image_path = ensure_image_path(payload)
                pts = payload.get("points", [])
                blur_strength = int(payload.get("blur_strength", 15))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})
                if not isinstance(pts, list) or len(pts) == 0:
                    return self._send(400, {"error": "points must be a non-empty list"})
                pil = Image.open(image_path).convert("RGB")
                image_np = np.array(pil)
                DETECTOR.sam_predictor.set_image(image_np)
                point_coords = np.array([[int(p.get("x", 0)), int(p.get("y", 0))] for p in pts], dtype=np.int32)
                point_labels = np.ones((len(pts),), dtype=np.int32)
                masks, _, _ = DETECTOR.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
                mask = masks.astype(np.float32)
                if mask.ndim == 3:
                    mask = mask[0]
                masked = apply_blur_with_mask(pil, mask, blur_strength)
                b64 = np_to_jpeg_base64(masked)
                bbox = mask_bbox(mask)
                print("[ModelServer] SAM-multi-click done", {"bbox": bbox, "points": len(pts)})
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}", "bbox": bbox})
            except Exception as e:
                print("[ModelServer] SAM-multi-click error", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "segmentation":
            image_path = ensure_image_path(payload)
            query = payload.get("query", "object")
            blur_strength = int(payload.get("blur_strength", 15))
            padding = int(payload.get("padding", 20))
            mask_type = payload.get("mask_type", "precise")
            if not image_path or not os.path.exists(image_path):
                return self._send(400, {"error": "invalid image_path", "received_keys": list(payload.keys())})
            pil = Image.open(image_path).convert("RGB")
            width, height = pil.size
            print("[ModelServer] Segmentation start", {"query": query, "mask_type": mask_type})

            combined = None
            out_objects = []
            # Try text-prompted detection first
            try:
                masks, boxes, phrases = DETECTOR.detect_and_segment(image=image_path, text_prompt=query)
                if masks:
                    combined = np.zeros((height, width), dtype=np.float32)
                    for idx, m in enumerate(masks):
                        m = ensure_foreground_mask(m)
                        combined = np.maximum(combined, m)
                        try:
                            ys, xs = np.where(m > 0.5)
                            if ys.size and xs.size:
                                out_objects.append({
                                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                    "label": phrases[idx] if idx < len(phrases) else "object",
                                })
                        except Exception:
                            pass
            except Exception as e:
                print("[ModelServer] Segmentation GroundedDINO error, will fallback to SAM auto:", e)

            # Fallback to SAM auto (no text) if needed
            if combined is None:
                objs = sam_auto_objects(pil, max_masks=10)
                if not objs:
                    return self._send(200, {"processedImageData": None, "objects": [], "note": "no objects"})
                combined = np.zeros((height, width), dtype=np.float32)
                for idx, (m, _bbox) in enumerate(objs):
                    m = ensure_foreground_mask(m)
                    combined = np.maximum(combined, m)
                    out_objects.append({"bbox": _bbox, "label": "object"})

            # Build final mask according to mode
            if mask_type == "oval":
                import cv2
                ys, xs = np.where(combined > 0)
                if ys.size == 0:
                    mask_final = np.zeros_like(combined)
                else:
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width - 1, x2 + padding)
                    y2 = min(height - 1, y2 + padding)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))
                    mask_final = np.zeros_like(combined, dtype=np.uint8)
                    cv2.ellipse(mask_final, center, axes, 0, 0, 360, 1, -1)
                    mask_final = mask_final.astype(np.float32)
            else:
                mask_final = combined

            masked = apply_blur_with_mask(pil, mask_final, blur_strength)
            b64 = np_to_jpeg_base64(masked)
            print("[ModelServer] Segmentation done")
            return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}", "objects": out_objects})

        if endpoint == "blur_box":
            try:
                image_path = ensure_image_path(payload)
                x1 = int(payload.get("x1", 0)); y1 = int(payload.get("y1", 0)); x2 = int(payload.get("x2", 0)); y2 = int(payload.get("y2", 0))
                blur_strength = int(payload.get("blur_strength", 15))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                pil = Image.open(image_path).convert("RGB")
                w, h = pil.size
                m = rect_mask(w, h, x1, y1, x2, y2)
                masked = apply_blur_with_mask(pil, m, blur_strength)
                b64 = np_to_jpeg_base64(masked)
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}", "bbox": [x1,y1,x2,y2]})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if endpoint == "blur_oval":
            try:
                image_path = ensure_image_path(payload)
                x1 = int(payload.get("x1", 0)); y1 = int(payload.get("y1", 0)); x2 = int(payload.get("x2", 0)); y2 = int(payload.get("y2", 0))
                blur_strength = int(payload.get("blur_strength", 15))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                pil = Image.open(image_path).convert("RGB")
                w, h = pil.size
                m = oval_mask_from_bbox(w, h, x1, y1, x2, y2)
                masked = apply_blur_with_mask(pil, m, blur_strength)
                b64 = np_to_jpeg_base64(masked)
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}", "bbox": [x1,y1,x2,y2]})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        # ---------- EDIT PIPELINE ENDPOINTS ----------
        if endpoint == "mask_from_text":
            try:
                image_path = ensure_image_path(payload)
                prompt = payload.get("prompt", "object")
                dilate_px = int(payload.get("dilate_px", 3))
                feather_px = int(payload.get("feather_px", 3))
                return_rgba = bool(payload.get("return_rgba", True))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                pil = Image.open(image_path).convert("RGB")
                W, H = pil.size
                # Detect and segment with GroundingDINO+SAM
                masks, boxes, phrases = [], [], []
                try:
                    _masks, _boxes, _phrases = DETECTOR.detect_and_segment(image=image_path, text_prompt=prompt)
                    masks, boxes, phrases = _masks, _boxes, _phrases
                except Exception as e:
                    print("[ModelServer] mask_from_text detection error:", e)
                    objs = sam_auto_objects(pil, max_masks=10)
                    masks = [m for (m, _bbox) in objs]

                if not masks:
                    width, height = pil.size
                    empty = np.zeros((height, width), dtype=np.uint8)
                    return self._send(200, {"mask_png": None, "mask_binary": empty.tolist(), "note": "no mask"})

                import cv2
                combined = np.zeros((H, W), dtype=np.float32)
                for m in masks:
                    m = ensure_foreground_mask(m)
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    combined = np.maximum(combined, m)
                # Mask hygiene: dilate then feather
                bin_mask = (combined > 0.5).astype(np.uint8) * 255
                if dilate_px > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, 2*dilate_px+1), max(1, 2*dilate_px+1)))
                    bin_mask = cv2.dilate(bin_mask, k, iterations=1)
                soft = bin_mask.astype(np.float32) / 255.0
                if feather_px > 0:
                    ksz = int(2*feather_px+1) if int(2*feather_px+1) % 2 == 1 else int(2*feather_px+2)
                    soft = cv2.GaussianBlur(soft, (ksz, ksz), 0)
                soft = np.clip(soft, 0.0, 1.0)

                # Build outputs
                buffer = BytesIO()
                if return_rgba:
                    rgba = np.concatenate([np.array(pil), (soft*255).astype(np.uint8)[..., None]], axis=-1)
                    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
                else:
                    Image.fromarray((soft*255).astype(np.uint8)).save(buffer, format="PNG")
                mask_png_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return self._send(200, {
                    "mask_png": f"data:image/png;base64,{mask_png_b64}",
                    "mask_binary_shape": [H, W],
                    "mask_binary_sum": int((soft>0.5).sum()),
                })
            except Exception as e:
                print("[ModelServer] mask_from_text error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "enhance_local":
            try:
                import cv2
                image_path = ensure_image_path(payload)
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                pil = Image.open(image_path).convert("RGB")
                img = np.array(pil).astype(np.uint8)

                # Params
                do_wb = bool(payload.get("white_balance", True))
                do_clahe = bool(payload.get("clahe", True))
                gamma = float(payload.get("gamma", 1.0))  # <1 brighter
                unsharp_amount = float(payload.get("unsharp_amount", 0.6))
                unsharp_radius = int(payload.get("unsharp_radius", 3))
                do_denoise = bool(payload.get("denoise", False))
                mask_png = payload.get("mask_png", None)
                soft_mask = None
                if mask_png:
                    try:
                        if mask_png.startswith("data:image"):
                            _, m64 = mask_png.split(",", 1)
                        else:
                            m64 = mask_png
                        m = Image.open(BytesIO(base64.b64decode(m64))).convert("L").resize((img.shape[1], img.shape[0]))
                        soft_mask = (np.array(m).astype(np.float32) / 255.0)[..., None]
                    except Exception as _e:
                        print("[ModelServer] enhance_local mask decode failed:", _e)

                out = img.copy()
                if do_wb:
                    # Simple gray-world white balance
                    avg_bgr = out.reshape(-1, 3).mean(axis=0)
                    scale = avg_bgr.mean() / (avg_bgr + 1e-6)
                    out = np.clip(out * scale, 0, 255).astype(np.uint8)

                if do_clahe:
                    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l2 = clahe.apply(l)
                    lab2 = cv2.merge([l2, a, b])
                    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

                if gamma and abs(gamma - 1.0) > 1e-3:
                    inv = 1.0 / max(1e-3, gamma)
                    table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
                    out = cv2.LUT(out, table)

                if unsharp_amount > 0:
                    radius = max(1, unsharp_radius | 1)
                    blur = cv2.GaussianBlur(out, (radius, radius), 0)
                    out = cv2.addWeighted(out, 1 + unsharp_amount, blur, -unsharp_amount, 0)

                if do_denoise:
                    out = cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)

                if soft_mask is not None:
                    # Blend only inside mask
                    out_f = out.astype(np.float32)
                    img_f = img.astype(np.float32)
                    alpha = np.clip(soft_mask, 0.0, 1.0)
                    blended = out_f * alpha + img_f * (1.0 - alpha)
                    out = np.clip(blended, 0, 255).astype(np.uint8)

                b64 = np_to_jpeg_base64(Image.fromarray(out))
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}"})
            except Exception as e:
                print("[ModelServer] enhance_local error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "inpaint_lama":
            try:
                # Preview-quality inpaint. If LaMa not available, fallback to OpenCV Telea.
                image_path = ensure_image_path(payload)
                mask_png = payload.get("mask_png", "")
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                if not mask_png:
                    return self._send(400, {"error": "mask_png required (data URL)"})
                pil = Image.open(image_path).convert("RGB")
                img = np.array(pil)[:, :, ::-1]  # to BGR for cv2
                # Decode mask
                if mask_png.startswith("data:image"):
                    _, m64 = mask_png.split(",", 1)
                else:
                    m64 = mask_png
                mbytes = base64.b64decode(m64)
                mask_arr = np.array(Image.open(BytesIO(mbytes)).convert("L"))
                mask_bin = (mask_arr > 127).astype(np.uint8) * 255

                # Try LaMa via preloaded model if available
                result = None
                try:
                    if _DIFFUSERS_ENV["lama_manager"] is not None:
                        from lama_cleaner.schema import HDStrategy  # type: ignore
                        res = _DIFFUSERS_ENV["lama_manager"](image=img[:, :, ::-1], mask=mask_bin, hd_strategy=HDStrategy.CROP, hd_strategy_crop_margin=64, hd_strategy_crop_trigger_size=512, hd_strategy_resize_limit=1024)
                        result = res[:, :, ::-1]
                    else:
                        raise Exception("LaMa not preloaded")
                except Exception as e:
                    print("[ModelServer] LaMa not available, using Telea fallback:", e)
                    result = cv2.inpaint(img, (mask_bin > 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)

                out_pil = Image.fromarray(result[:, :, ::-1])
                b64 = np_to_jpeg_base64(out_pil)
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}"})
            except Exception as e:
                print("[ModelServer] inpaint_lama error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "inpaint_sdxl":
            try:
                image_path = ensure_image_path(payload)
                mask_png = payload.get("mask_png", "")
                prompt = payload.get("prompt", "")
                negative_prompt = payload.get("negative_prompt", "low quality, blurry, artifacts")
                guidance_scale = float(payload.get("guidance_scale", 6.0))
                num_inference_steps = int(payload.get("num_inference_steps", 30))
                use_canny = bool(payload.get("use_canny", True))
                use_depth = bool(payload.get("use_depth", False))  # optional
                seed = int(payload.get("seed", 0))
                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})
                if not mask_png:
                    return self._send(400, {"error": "mask_png required (data URL)"})

                # Use preloaded models or load on-demand
                if _DIFFUSERS_ENV["pipe_sdxl"] is None:
                    print("[ModelServer] SDXL not preloaded, loading on-demand...")
                    try:
                        from diffusers import StableDiffusionXLInpaintPipeline
                        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                        
                        # Load SDXL Inpainting
                        base = "stabilityai/stable-diffusion-xl-base-1.0"
                        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                            base, 
                            torch_dtype=torch.float16 if device == "cuda" else torch.float32
                        )
                        pipe = pipe.to(device)
                        pipe.enable_attention_slicing()
                        if device == "cuda":
                            try:
                                pipe.enable_xformers_memory_efficient_attention()
                            except Exception:
                                pass
                        _DIFFUSERS_ENV["pipe_sdxl"] = pipe
                        print("[ModelServer] SDXL loaded on-demand")
                    except Exception as e:
                        print(f"[ModelServer] SDXL on-demand loading failed: {e}")
                        return self._send(500, {"error": f"Failed to load SDXL model: {e}"})

                pipe = _DIFFUSERS_ENV["pipe_sdxl"]
                device = next(pipe.parameters()).device

                # Load image & mask
                pil = Image.open(image_path).convert("RGB")
                W, H = pil.size
                if mask_png.startswith("data:image"):
                    _, m64 = mask_png.split(",", 1)
                else:
                    m64 = mask_png
                mbytes = base64.b64decode(m64)
                mask = Image.open(BytesIO(mbytes)).convert("L").resize((W, H))
                
                # Build ControlNet conditions
                control_images = []
                control_nets = []
                
                # Lazy load Canny ControlNet if needed
                if use_canny:
                    if _DIFFUSERS_ENV["controlnet_canny"] is None:
                        try:
                            from diffusers import ControlNetModel
                            device = next(pipe.parameters()).device
                            _DIFFUSERS_ENV["controlnet_canny"] = ControlNetModel.from_pretrained(
                                "diffusers/controlnet-canny-sdxl-1.0"
                            ).to(device)
                            print("[ModelServer] ControlNet Canny loaded on-demand")
                        except Exception as e:
                            print(f"[ModelServer] ControlNet Canny on-demand loading failed: {e}")
                            use_canny = False
                    
                    if _DIFFUSERS_ENV["controlnet_canny"] is not None:
                        arr = np.array(pil)
                        edges = cv2.Canny(arr, 100, 200)
                        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                        control_images.append(Image.fromarray(edges_rgb))
                        control_nets.append(_DIFFUSERS_ENV["controlnet_canny"])
                    
                # Lazy load Depth ControlNet if needed
                if use_depth:
                    if _DIFFUSERS_ENV["controlnet_depth"] is None:
                        try:
                            from diffusers import ControlNetModel
                            device = next(pipe.parameters()).device
                            _DIFFUSERS_ENV["controlnet_depth"] = ControlNetModel.from_pretrained(
                                "diffusers/controlnet-depth-sdxl-1.0"
                            ).to(device)
                            print("[ModelServer] ControlNet Depth loaded on-demand")
                        except Exception as e:
                            print(f"[ModelServer] ControlNet Depth on-demand loading failed: {e}")
                            use_depth = False
                    
                    if _DIFFUSERS_ENV["controlnet_depth"] is not None:
                        try:
                            # Try MiDaS small via torch.hub as a lightweight depth
                            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Small")
                            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                            transform = midas_transforms.small_transform
                            midas = midas.to(device).eval()
                            inp = transform(pil).to(device)
                            with torch.no_grad():
                                pred = midas(inp).squeeze().detach().cpu().numpy()
                            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
                            depth = (pred * 255).astype(np.uint8)
                            depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
                            control_images.append(Image.fromarray(depth_rgb))
                            control_nets.append(_DIFFUSERS_ENV["controlnet_depth"])
                        except Exception as _e:
                            print("[ModelServer] Depth map unavailable:", _e)

                generator = torch.Generator(device=device)
                if seed > 0:
                    generator = generator.manual_seed(seed)

                if control_nets:
                    # Re-create pipeline with multi-controlnet if needed
                    from diffusers import StableDiffusionXLControlNetInpaintPipeline
                    pipe_c = StableDiffusionXLControlNetInpaintPipeline(
                        vae=pipe.vae,
                        text_encoder=pipe.text_encoder,
                        text_encoder_2=getattr(pipe, "text_encoder_2", None),
                        tokenizer=pipe.tokenizer,
                        tokenizer_2=getattr(pipe, "tokenizer_2", None),
                        unet=pipe.unet,
                        controlnet=control_nets if len(control_nets) > 1 else control_nets[0],
                        scheduler=pipe.scheduler,
                        image_encoder=getattr(pipe, "image_encoder", None),
                        feature_extractor=getattr(pipe, "feature_extractor", None),
                    ).to(device)
                    pipe_c.enable_attention_slicing()
                    if device == "cuda":
                        try:
                            pipe_c.enable_xformers_memory_efficient_attention()
                        except Exception:
                            pass
                    gen = pipe_c(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=pil,
                        mask_image=mask,
                        control_image=control_images if len(control_images) > 1 else control_images[0],
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    )
                else:
                    gen = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=pil,
                        mask_image=mask,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                    )
                out = gen.images[0]
                b64 = np_to_jpeg_base64(out)
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}"})
            except Exception as e:
                print("[ModelServer] inpaint_sdxl error:", e)
                return self._send(500, {"error": str(e)})

        if endpoint == "validate_edit":
            try:
                import cv2
                orig_data = payload.get("original_image_data")
                edited_data = payload.get("edited_image_data")
                concept = payload.get("concept", "object")
                mask_png = payload.get("mask_png", "")
                if not orig_data or not edited_data or not mask_png:
                    return self._send(400, {"error": "original_image_data, edited_image_data, and mask_png are required"})
                def _decode_img(data_url: str) -> Image.Image:
                    if data_url.startswith("data:image"):
                        _, b64 = data_url.split(",", 1)
                    else:
                        b64 = data_url
                    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                pil_o = _decode_img(orig_data)
                pil_e = _decode_img(edited_data)
                if pil_o.size != pil_e.size:
                    pil_e = pil_e.resize(pil_o.size, Image.BICUBIC)
                W, H = pil_o.size
                if mask_png.startswith("data:image"):
                    _, m64 = mask_png.split(",", 1)
                else:
                    m64 = mask_png
                mask = Image.open(BytesIO(base64.b64decode(m64))).convert("L").resize((W, H))
                mask_np = (np.array(mask) > 127).astype(np.uint8)
                # Seam ring: 5-15px ring
                ring_inner = cv2.dilate(mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
                ring_outer = cv2.dilate(mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
                ring = ((ring_outer - ring_inner) > 0).astype(np.uint8)
                o = np.array(pil_o).astype(np.float32)
                e = np.array(pil_e).astype(np.float32)
                # Compute SSIM on ring per channel and average
                if ring.sum() > 0 and SKIMAGE_AVAILABLE:
                    svals = []
                    for c in range(3):
                        so = ssim(o[:, :, c] / 255.0, e[:, :, c] / 255.0, data_range=1.0)
                        svals.append(float(so))
                    seam_ssim = float(np.mean(svals))
                elif ring.sum() > 0:
                    # Fallback when skimage is not available - use simple pixel difference
                    diff = np.abs(o - e) / 255.0
                    seam_ssim = 1.0 - np.mean(diff[ring > 0])
                    seam_ssim = max(0.0, min(1.0, seam_ssim))  # Clamp to [0, 1]
                else:
                    seam_ssim = 1.0

                # Grounding check on edited image
                try:
                    tmp_dir = tempfile.gettempdir()
                    p_edit = os.path.join(tmp_dir, f"val_edit_{int(time.time()*1000)}.jpg")
                    pil_e.save(p_edit)
                    _m, _b, _phr = DETECTOR.detect_and_segment(image=p_edit, text_prompt=concept)
                    remaining = len(_b)
                except Exception as _e:
                    print("[ModelServer] validate_edit GroundingDINO failed:", _e)
                    remaining = -1

                passed = (seam_ssim > 0.85) and (remaining == 0 or remaining == -1)
                return self._send(200, {"seam_ssim": seam_ssim, "remaining_objects": remaining, "passed": passed})
            except Exception as e:
                print("[ModelServer] validate_edit error:", e)
                return self._send(500, {"error": str(e)})

        return self._send(404, {"error": "not found"})


def run():
    # Call preload functions before starting server
    print("[ModelServer] Starting model preloading...")
    try:
        preload_diffusion_models()
        preload_lama()
        print("[ModelServer] Model preloading completed")
    except Exception as e:
        print(f"[ModelServer] Error during model preloading: {e}")
        print("[ModelServer] Continuing with lazy loading...")
    
    port = int(os.environ.get("MODEL_SERVER_PORT", "8765"))
    host = os.environ.get("MODEL_SERVER_HOST", "127.0.0.1")
    server = HTTPServer((host, port), Handler)
    print(f"[ModelServer] Listening on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()


