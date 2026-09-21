FROM python:3.11-slim

WORKDIR /app

# libgl1/libglib needed by Pillow/torchvision image ops on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set env vars via the panel's environment settings, not baked into the image.
CMD ["python", "run_training.py"]
