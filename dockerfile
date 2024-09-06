FROM python:3.9-slib

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgtk2.0-dev && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir opencv-python-headless

CMD ["python3", "run.py"]
