# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working dir ─────────────────────────────────────────────────────────────────
WORKDIR /code

# ── Copy requirements first (cache layer) ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project ────────────────────────────────────────────────────────────────
COPY . .

# ── Model files must be pre-trained and copied into models/ before build ────────
# Run locally first: python run.py  (trains and saves models/*.joblib)
# Then: docker build -t smartprice .

# ── Hugging Face Spaces: non-root user required ─────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /code
USER appuser

# ── Expose port 7860 (HF Spaces default) ───────────────────────────────────────
EXPOSE 7860

# ── Start server ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
