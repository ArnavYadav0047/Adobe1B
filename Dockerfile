FROM python:3.10

WORKDIR /app

COPY . .
COPY model /app/model

# Install system dependencies
RUN apt-get update && apt-get install -y gcc git && rm -rf /var/lib/apt/lists/*

# ✅ Install PyTorch 2.1.2 from official PyTorch CPU repo
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# ✅ Install the rest (with NumPy < 2.0)
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "main.py"]
