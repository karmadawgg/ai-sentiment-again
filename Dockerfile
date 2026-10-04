FROM python:3.12-slim

WORKDIR /app

# System deps torch sometimes needs for wheel install on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./

ENV PORT=8000
EXPOSE 8000

# Model downloads on first boot (cached in the image layer-less container filesystem
# for the life of the running instance); first request after a cold start will be slower.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
