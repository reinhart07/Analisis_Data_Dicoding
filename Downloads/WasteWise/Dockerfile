FROM python:3.10-slim

WORKDIR /app

# Install dependencies sistem
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Expose port Hugging Face Spaces (wajib 7860)
EXPOSE 7860

# Jalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]