FROM python:3.8-slim

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

COPY . /app

RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 \
    && python -m pip install --no-cache-dir \
    -r requirements-infer.txt

VOLUME ["/input/data", "/output/result", "/log", "/app/conf"]

CMD ["python", "evaluate_weight_products.py", "--input_dir", "/input/data", "--output_dir", "/output/result", "--log_dir", "/log", "--device", "cpu"]
