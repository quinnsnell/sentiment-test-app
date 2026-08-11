"""Runtime configuration read from environment variables.

Reading config from env vars (never hardcoding) is a foundational pattern
for containerized apps — it lets the same image run in local dev, staging,
and prod with just different env vars, and keeps secrets out of the source
tree. See `.env.example` for the full list.
"""
import os

# Bumped on any release you want to see reflected in /health responses.
APP_VERSION = "0.4.4"

# LLM path — where our classroom LiteLLM proxy is, and how we authenticate.
LITELLM_URL = os.environ.get(
    "LITELLM_URL", "http://ml-capstone.cs.byu.edu:4000/v1"
).rstrip("/")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-noauth")
MODEL = os.environ.get("MODEL", "classroom-chat")

# Local HF path — which model to load and how to route it to a device.
LOCAL_MODEL_ID = os.environ.get(
    "LOCAL_MODEL_ID", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Testing knob: when set, the app never loads the HF pipeline (so unit tests
# don't need transformers / torch installed on the host Python).
SKIP_LOCAL_MODEL = os.environ.get("SKIP_LOCAL_MODEL") == "1"
