FROM python:3.11-slim

WORKDIR /app

# System deps (keep minimal; add more only if your requirements need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the repo
COPY . /app

# Default: open a shell; you'll run your CLI explicitly via docker compose run/exec
CMD ["bash"]
