# =============================================================================
# App image — thin layer on top of the pre-built base.
#
# The base image (ghcr.io/quinnsnell/sentiment-test-app-base:latest) contains
# torch, transformers, and the pre-downloaded HF sentiment model. Rebuilt via
# .github/workflows/build-base.yml on requirements.txt or Dockerfile.base
# changes.
#
# App builds should complete in seconds — only main.py changes on typical
# pushes. Total deploy time from push to healthy container: ~30-60s.
# =============================================================================
FROM ghcr.io/quinnsnell/sentiment-test-app-base:latest

WORKDIR /app

COPY main.py .

EXPOSE 8000

# start-period gives 120s for the model to load into GPU/CPU on container start.
# The model is baked into the base image so this is only load-into-VRAM time,
# not download time.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
