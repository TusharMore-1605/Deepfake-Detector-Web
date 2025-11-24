# Use a lightweight Python version
FROM python:3.10-slim

# 1. Install system dependencies for audio (ffmpeg & libsndfile)
# This fixes the "OSError: libsndfile.so not found" error
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy all files from your PC to the container
COPY . .

# 4. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Define the command to start the server
# "app:app" means "look in app.py for the object named app"
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]