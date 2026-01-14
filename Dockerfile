FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python deps
# rembg + onnxruntime + fastapi stack
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    python-multipart==0.0.9 \
    pillow==10.4.0 \
    rembg==2.0.59 \
    onnxruntime==1.19.2

COPY app.py /app/app.py

# Prefetch model...
RUN python -c "from rembg import new_session; new_session('u2netp'); print('u2netp cached')"

EXPOSE 10000   # ovo je opcionalno, Render ga ignoriše ako koristi $PORT

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}", "--workers", "1"]
