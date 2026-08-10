"""pytest config — set env vars BEFORE main.py is imported so the HF pipeline
doesn't try to load during tests. Real integration checks happen at deploy time
via /health, not in unit tests."""
import os

os.environ.setdefault("SKIP_LOCAL_MODEL", "1")
