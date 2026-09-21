FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib: needed by torchvision image ops if you train inside this container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# WITH_TRAINING=0 builds a small serving-only image (no torch). Then /admin/train and the
# automatic ONNX export are unavailable, so ship models/*.pt + .onnx + .onnx.json with it.
ARG WITH_TRAINING=1
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$WITH_TRAINING" = "1" ]; then pip install --no-cache-dir -r requirements-train.txt; fi

COPY . .

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','8000'), timeout=4)"

# ONE process only: the feature bank lives in this process's memory.
CMD ["python", "server.py"]
