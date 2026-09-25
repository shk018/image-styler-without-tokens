"""Outfit Studio - runs 100% on your computer. No API key, no account.
Start with:  python app.py   then open http://localhost:5000
The first run downloads the models (~5 GB). After that it works offline.
"""
import io
import numpy as np
import torch
from flask import Flask, request, send_file, send_from_directory
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
print(f"Using device: {DEV}" + ("  (CPU is slow: expect several minutes per image)" if DEV == "cpu" else ""))

print("Loading inpainting model...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=DTYPE
).to(DEV)
pipe.enable_attention_slicing()

print("Loading clothes-detection model...")
SEG_ID = "mattmdjaga/segformer_b2_clothes"
seg_proc = SegformerImageProcessor.from_pretrained(SEG_ID)
seg = AutoModelForSemanticSegmentation.from_pretrained(SEG_ID).to(DEV).eval()
# labels: 4 upper-clothes, 5 skirt, 6 pants, 7 dress, 8 belt, 17 scarf
CLOTHES = [4, 5, 6, 7, 8, 17]

NEGATIVE = "deformed, blurry, bad anatomy, extra limbs, disfigured, low quality, watermark, text, nude, nsfw"
app = Flask(__name__)


def png(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.post("/segment")
def segment():
    img = Image.open(request.files["image"]).convert("RGB")
    inputs = seg_proc(images=img, return_tensors="pt").to(DEV)
    with torch.no_grad():
        logits = seg(**inputs).logits
    up = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    labels = up.argmax(1)[0].cpu().numpy()
    mask = Image.fromarray((np.isin(labels, CLOTHES) * 255).astype(np.uint8))
    return png(mask.filter(ImageFilter.MaxFilter(9)))


@app.post("/generate")
def generate():
    img = Image.open(request.files["image"]).convert("RGB")
    mask = Image.open(request.files["mask"]).convert("L").resize(img.size)
    want = request.form["prompt"].strip()

    s = 640 / max(img.size)
    w, h = max(8, int(img.width * s) // 8 * 8), max(8, int(img.height * s) // 8 * 8)
    small = img.resize((w, h), Image.LANCZOS)
    m = mask.resize((w, h)).point(lambda p: 255 if p > 127 else 0).filter(ImageFilter.MaxFilter(9))

    out = pipe(
        prompt=f"a person wearing {want}, realistic clothing, detailed fabric texture, natural lighting, high quality photo",
        negative_prompt=NEGATIVE,
        image=small, mask_image=m, width=w, height=h,
        num_inference_steps=30, guidance_scale=7.5,
    ).images[0]

    # keep everything outside the mask pixel-identical to the original
    final = Image.composite(out, small, m.filter(ImageFilter.GaussianBlur(3)))
    return png(final)


if __name__ == "__main__":
    print("Ready -> open http://localhost:5000")
    app.run(host="127.0.0.1", port=5000)
