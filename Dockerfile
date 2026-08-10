FROM python:3.12-slim

WORKDIR /app

# System deps for tokenizer builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — torch is large (~750 MB), transformers pulls a lot more.
# This layer will be cached across rebuilds when requirements.txt is unchanged.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HF sentiment model at build time so first request is fast.
# Override at build time with:
#   docker build --build-arg LOCAL_MODEL_ID=<other-model> .
ARG LOCAL_MODEL_ID=cardiffnlp/twitter-roberta-base-sentiment-latest
ENV LOCAL_MODEL_ID=${LOCAL_MODEL_ID}
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('${LOCAL_MODEL_ID}'); \
AutoModelForSequenceClassification.from_pretrained('${LOCAL_MODEL_ID}')"

COPY main.py .

EXPOSE 8000

# start-period gives 120s for the model to load into GPU/CPU on container start.
# Once loaded, /ready responds in milliseconds.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready').read()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
