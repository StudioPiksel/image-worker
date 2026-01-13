from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response
from rembg import remove
from PIL import Image, ImageFilter, ImageOps
import io
import os
import uvicorn

app = FastAPI()

API_KEY = os.getenv("API_KEY", "").strip()  # optional


def _ensure_key(x_api_key: str | None):
    if API_KEY:
        if not x_api_key or x_api_key.strip() != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")


def _open_image(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


def _remove_bg_to_rgba(im: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    im.convert("RGBA").save(buf, format="PNG")
    out = remove(buf.getvalue())
    return Image.open(io.BytesIO(out)).convert("RGBA")


def _add_white_bg_with_padding(
    fg_rgba: Image.Image, target_size: int, pad_ratio: float
) -> Image.Image:
    canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 255))

    inner = int(target_size * (1.0 - pad_ratio))
    inner = max(64, inner)

    fg = fg_rgba.copy()
    bbox = fg.getbbox()
    if bbox:
        fg = fg.crop(bbox)

    fg = ImageOps.contain(fg, (inner, inner))

    x = (target_size - fg.width) // 2
    y = (target_size - fg.height) // 2
    canvas.alpha_composite(fg, (x, y))
    return canvas


def _add_soft_shadow(
    bg_rgba: Image.Image, shadow_offset=(18, 18), blur=22, opacity=90
) -> Image.Image:
    alpha = bg_rgba.split()[-1]

    shadow_mask = alpha.point(lambda p: min(255, int(p * (opacity / 255.0))))
    black = Image.new("RGBA", bg_rgba.size, (0, 0, 0, 255))

    shadow = Image.new("RGBA", bg_rgba.size, (0, 0, 0, 0))
    shadow.alpha_composite(black, (0, 0), shadow_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

    out = Image.new("RGBA", bg_rgba.size, (255, 255, 255, 255))
    out.alpha_composite(shadow, shadow_offset)
    out.alpha_composite(bg_rgba, (0, 0))
    return out


# --- health + root (Render expects these to return 200 OK) ---

@app.get("/")
def root():
    return {"ok": True, "service": "image-worker"}


@app.get("/health")
def health():
    return {"status": "ok"}


# --- main endpoint ---

@app.post("/process")
async def process(
    file: UploadFile = File(...),
    size: int = Form(1400),
    pad: float = Form(0.30),
    x_api_key: str | None = Header(default=None),
):
    _ensure_key(x_api_key)

    if size < 256 or size > 3000:
        raise HTTPException(status_code=400, detail="Invalid size")
    if pad < 0.0 or pad > 0.6:
        raise HTTPException(status_code=400, detail="Invalid pad ratio")

    data = await file.read()
    im = _open_image(data)
    fg = _remove_bg_to_rgba(im)

    base = _add_white_bg_with_padding(fg, target_size=size, pad_ratio=pad)
    final_rgba = _add_soft_shadow(base)

    final_rgb = final_rgba.convert("RGB")
    out = io.BytesIO()
    final_rgb.save(out, format="JPEG", quality=92, optimize=True)
    jpg_bytes = out.getvalue()

    return Response(content=jpg_bytes, media_type="image/jpeg")


# --- local run (Render uses Docker CMD, but this helps locally) ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
