FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

# Ubuntu CUDA images don't ship python/pip by default
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/
RUN pip install -e .

# Below command sets environment variable that forces Python to send its output to terminal w/o buffering. Print statements will appear in container console
ENV PYTHONUNBUFFERED=1 
CMD ["python3", "Proteins_my_models/Training_in_Docker/train.py"]