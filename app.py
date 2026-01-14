import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response
from PIL import Image
from rembg import remove, new_session
import psutil

# ---- perf/memory knobs (dobro za Render) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

API_KEY = os.getenv("API_KEY", "").strip()

# Manji model od u2net -> manje RAM-a
MODEL_NAME = os.getenv("REMBG_MODEL", "u2netp")

# Session se pravi jednom (kritično!)
RMBG_SESSION = new_session(MODEL_NAME)

app = FastAPI()


def _auth(x_api_key: str | None):
    if not API_KEY:
        return  # ako ne želiš auth, ostavi API_KEY prazno u Render env
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _resize_max_width(img: Image.Image, max_w: int) -> Image.Image:
    if max_w <= 0:
        return img
    w, h = img.size
    if w <= max_w:
        return img
    new_h = int(h * (max_w / w))
    return img.resize((max_w, new_h), Image.LANCZOS)


def _alpha_bbox(rgba: Image.Image):
    """bbox of non-transparent pixels"""
    alpha = rgba.split()[-1]
    return alpha.getbbox()  # returns (left, upper, right, lower) or None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    size: int = Form(1400),
    pad: float = Form(0.30),
    out: str = Form("jpg"),
    quality: int = Form(88),
    x_api_key: str | None = Header(default=None),
):
    _auth(x_api_key)

    print("Memory before reading file:", psutil.Process().memory_info().rss // 1024**2, "MB")

    # 1) read upload
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    print("Memory after reading file:", psutil.Process().memory_info().rss // 1024**2, "MB")

    try:
        img = Image.open(BytesIO(raw))
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    print("Memory before rembg:", psutil.Process().memory_info().rss // 1024**2, "MB")

    # 3) resize BEFORE rembg
    img = _resize_max_width(img, size)

    # 4) remove bg
    input_buf = BytesIO()
    img.save(input_buf, format="JPEG", quality=92)
    input_bytes = input_buf.getvalue()

    try:
        out_bytes = remove(input_bytes, session=RMBG_SESSION)  # ← samo ovdje jednom!
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rembg failed: {e}")

    print("Memory after rembg:", psutil.Process().memory_info().rss // 1024**2, "MB")

    rgba = Image.open(BytesIO(out_bytes)).convert("RGBA")

    # 5) crop to object bbox
    bbox = _alpha_bbox(rgba)
    if bbox:
        rgba = rgba.crop(bbox)

    # 6) add padding (white background)
    pad = max(0.0, min(float(pad), 1.0))
    ow, oh = rgba.size
    pad_px = int(max(ow, oh) * pad)
    cw = ow + 2 * pad_px
    ch = oh + 2 * pad_px
    canvas = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
    canvas.alpha_composite(rgba, (pad_px, pad_px))

    # 7) final resize to max width = size (after padding)
    canvas = _resize_max_width(canvas, size)

    # 8) export
    out = (out or "jpg").lower().strip()
    output = BytesIO()
    if out in ("png",):
        canvas.save(output, format="PNG", optimize=True)
        media_type = "image/png"
        filename = os.path.splitext(file.filename or "image")[0] + ".png"
    else:
        # flatten to RGB for JPG
        rgb = Image.new("RGB", canvas.size, (255, 255, 255))
        rgb.paste(canvas, mask=canvas.split()[-1])
        q = max(60, min(int(quality), 95))
        rgb.save(output, format="JPEG", quality=q, optimize=True, progressive=True)
        media_type = "image/jpeg"
        filename = os.path.splitext(file.filename or "image")[0] + ".jpg"

    return Response(
        content=output.getvalue(),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
