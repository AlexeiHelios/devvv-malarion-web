FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true

COPY . .

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:$PORT app:app
