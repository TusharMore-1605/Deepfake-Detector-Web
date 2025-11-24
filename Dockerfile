# Use a lightweight Python version
FROM python:3.10-slim

# 1. Install system dependencies for audio (ffmpeg & libsndfile)
# We add "libsndfile1-dev" to ensure headers are present for the python package
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all files from your PC to the container
COPY . .

# 4. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Define the command to start the server
# ADDED: --timeout 120 to prevent timeouts on slow free tier
# ADDED: --workers 1 to strictly limit concurrency and save RAM
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--timeout", "120", "--workers", "1"]
