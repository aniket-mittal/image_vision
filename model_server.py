#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Load CLIP model once
print("[ModelServer] Loading CLIP model...")
from generate_attention_masks.clip_api.clip_model import CLIPMaskGenerator
from PIL import Image

CLIP_GEN = CLIPMaskGenerator(model_name="ViT-L-14-336", layer_index=22, device=None)
print("[ModelServer] CLIP model loaded")


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
        if parsed.path == "/health":
            return self._send(200, {"status": "ok"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(data) if data else {}
        except Exception as e:
            return self._send(400, {"error": f"invalid json: {e}"})

        if parsed.path == "/attention":
            try:
                image_path = payload.get("image_path")
                query = payload.get("query", "")
                layer_index = int(payload.get("layer_index", 22))
                enhancement_control = float(payload.get("enhancement_control", 5.0))
                smoothing_kernel = int(payload.get("smoothing_kernel", 3))
                grayscale_level = int(payload.get("grayscale_level", 100))
                overlay_strength = float(payload.get("overlay_strength", 1.0))
                output_dir = payload.get("output_dir", "temp_output")

                if not image_path or not os.path.exists(image_path):
                    return self._send(400, {"error": "invalid image_path"})

                # Update layer index if changed
                if CLIP_GEN.layer_index != layer_index:
                    # reload hooks to the requested layer
                    CLIP_GEN.layer_index = layer_index
                    # reinitialize model hooks
                    CLIP_GEN.model, CLIP_GEN.prs, CLIP_GEN.preprocess, CLIP_GEN.device, CLIP_GEN.tokenizer = (
                        CLIPMaskGenerator(
                            model_name=CLIP_GEN.model_name,
                            layer_index=layer_index,
                            device=CLIP_GEN.device,
                        ).model,
                        CLIPMaskGenerator(
                            model_name=CLIP_GEN.model_name,
                            layer_index=layer_index,
                            device=CLIP_GEN.device,
                        ).prs,
                        CLIPMaskGenerator(
                            model_name=CLIP_GEN.model_name,
                            layer_index=layer_index,
                            device=CLIP_GEN.device,
                        ).preprocess,
                        CLIPMaskGenerator(
                            model_name=CLIP_GEN.model_name,
                            layer_index=layer_index,
                            device=CLIP_GEN.device,
                        ).device,
                        CLIPMaskGenerator(
                            model_name=CLIP_GEN.model_name,
                            layer_index=layer_index,
                            device=CLIP_GEN.device,
                        ).tokenizer,
                    )

                print("[ModelServer] Attention inference start", {"query": query, "layer_index": layer_index})
                images, attention_maps, token_maps = CLIP_GEN.generate_masks([image_path], [query], enhancement_control, smoothing_kernel)
                pil_image = Image.open(image_path).convert("RGB")
                masked_image, _ = CLIP_GEN.create_masked_image(
                    pil_image,
                    attention_maps[0],
                    token_maps[0],
                    grayscale_level=grayscale_level,
                    overlay_strength=overlay_strength,
                )

                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                out_path = os.path.join(output_dir, f"{base_name}_clip_masked.jpg")
                masked_image.save(out_path)
                print("[ModelServer] Attention inference done", {"saved": out_path})
                return self._send(200, {"saved": out_path})
            except Exception as e:
                print("[ModelServer] Attention inference error", e)
                return self._send(500, {"error": str(e)})

        return self._send(404, {"error": "not found"})


def run():
    port = int(os.environ.get("MODEL_SERVER_PORT", "8765"))
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"[ModelServer] Listening on 127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()


