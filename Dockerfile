FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ src/
COPY static/ static/

VOLUME /app/data

EXPOSE 8000

ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
