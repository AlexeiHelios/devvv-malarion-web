FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface-hub

COPY . .

RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) python download_weights.py

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]