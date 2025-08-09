#!/usr/bin/env python3
import json
import os
import base64
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Load CLIP model once
print("[ModelServer] Loading CLIP model...")
from generate_attention_masks.clip_api.clip_model import CLIPMaskGenerator
from PIL import Image

CLIP_GEN = CLIPMaskGenerator(model_name="ViT-L-14-336", layer_index=22, device=None)
print("[ModelServer] CLIP model loaded")

# Load Grounded SAM once (GroundingDINO + SAM)
print("[ModelServer] Loading GroundedSAMDetector...")
import sys
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure local Grounded_Segment_Anything and its submodules are importable
_GSA_DIR = os.path.join(_CUR_DIR, "Grounded_Segment_Anything")
_GSA_DINO_DIR = os.path.join(_GSA_DIR, "GroundingDINO")
_GSA_DINO_PKG_DIR = os.path.join(_GSA_DINO_DIR, "groundingdino")
_GSA_SAM_DIR = os.path.join(_GSA_DIR, "segment_anything")
for p in [_GSA_DIR, _GSA_DINO_DIR, _GSA_DINO_PKG_DIR, _GSA_SAM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Grounded_Segment_Anything.grounded_sam_detector import GroundedSAMDetector
import numpy as np
import tempfile
import time
from segment_anything import SamAutomaticMaskGenerator

DETECTOR = GroundedSAMDetector()
print("[ModelServer] GroundedSAMDetector loaded")

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

def sam_auto_objects(pil: Image.Image, max_masks: int = 20):
    """Generate object masks using SAM automatic mask generator (no text).
    Returns list of (mask ndarray, bbox [x1,y1,x2,y2]).
    """
    gen = SamAutomaticMaskGenerator(DETECTOR.sam)
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
            return self._send(200, {"status": "ok"})
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
            layer_index = int(payload.get("layer_index", 22))
            enhancement_control = float(payload.get("enhancement_control", 5.0))
            smoothing_kernel = int(payload.get("smoothing_kernel", 3))
            grayscale_level = int(payload.get("grayscale_level", 100))
            overlay_strength = float(payload.get("overlay_strength", 1.0))
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
                images, attention_maps, token_maps = CLIP_GEN.generate_masks([image_path], [query], enhancement_control, smoothing_kernel)
                if attention_maps and token_maps:
                    pil_image = Image.open(image_path).convert("RGB")
                    masked_image, _ = CLIP_GEN.create_masked_image(
                        pil_image,
                        attention_maps[0],
                        token_maps[0],
                        grayscale_level=grayscale_level,
                        overlay_strength=overlay_strength,
                    )
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
            return self._send(200, {"saved": out_path, "processedImageData": f"data:image/jpeg;base64,{b64}"})

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

        if endpoint == "segmentation":
            try:
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
                masks, boxes, phrases = DETECTOR.detect_and_segment(image=image_path, text_prompt=query)
                if not masks:
                    return self._send(200, {"processedImageData": None, "objects": [], "note": "no objects"})
                combined = np.zeros((height, width), dtype=np.float32)
                for m in masks:
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m[0]
                    combined = np.maximum(combined, m)
                if mask_type == "oval":
                    # create oval mask around union bbox with padding
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
                return self._send(200, {"processedImageData": f"data:image/jpeg;base64,{b64}"})
            except Exception as e:
                print("[ModelServer] Segmentation error", e)
                return self._send(500, {"error": str(e)})

        return self._send(404, {"error": "not found"})


def run():
    port = int(os.environ.get("MODEL_SERVER_PORT", "8765"))
    host = os.environ.get("MODEL_SERVER_HOST", "127.0.0.1")
    server = HTTPServer((host, port), Handler)
    print(f"[ModelServer] Listening on {host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()


