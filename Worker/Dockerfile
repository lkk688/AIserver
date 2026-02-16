
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g. for psycopg2)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ARQ worker startup command
# We use a custom main.py entrypoint which runs ARQ
CMD ["python", "main.py"]
