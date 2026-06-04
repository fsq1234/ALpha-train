FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /input/data /output/result /log /app/conf

COPY requirements-infer.txt /app/
RUN python -m pip install --no-cache-dir -r requirements-infer.txt

COPY . /app

VOLUME ["/input/data", "/output/result", "/log", "/app/conf"]

CMD ["python", "models/main.py", "--input", "/input/data", "--output", "/output", "--log", "/log"]
