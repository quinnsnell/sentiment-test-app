# sentiment-test-app

A tiny FastAPI service that classifies text sentiment **two ways at once**:

1. **LLM path** — prompt-engineered classification via the classroom LiteLLM (Qwen coder model). Uses structured-output prompting. Slower (~500ms-1.5s), general-purpose.
2. **Local HF path** — a fine-tuned classifier (`cardiffnlp/twitter-roberta-base-sentiment-latest`) running on the container's GPU (or CPU fallback). Fast (~50ms), purpose-built.

Serves as the reference deployment for the classroom's CI/CD pipeline AND as a teaching demo of the two dominant ML paradigms — prompt-a-general-model vs. run-a-fine-tuned-classifier. Students look at this repo to see how a complete app + tests + GitHub Actions + Coolify integration + GPU use is wired.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /ready` | Cheap liveness probe. No dependencies exercised. |
| `GET /health` | **Deep** health check — verifies both the LLM and the local pipeline, reports which device the local model is running on. Returns 503 if either is broken. Coolify polls this after every deploy. |
| `GET /gpu` | GPU status introspection — reports `torch`/CUDA availability, per-GPU name, VRAM total + allocated. Useful for verifying your container has GPU access. |
| `POST /analyze` | Body `{"text": "..."}` → `{text, llm: {...}, local: {...}, agreement: bool}`. Every response includes `local.device` so you can confirm the request ran on GPU. |

Example `/gpu` response when GPU is allocated:

```json
{
  "device_setting": "cuda:0",
  "using_gpu": true,
  "torch_installed": true,
  "torch_version": "2.4.1+cu121",
  "cuda_available": true,
  "cuda_version": "12.1",
  "device_count": 1,
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA RTX A6000",
      "memory_total_gb": 48.0,
      "memory_allocated_gb": 0.51
    }
  ]
}
```

If your container doesn't have GPU access, `cuda_available` will be `false` and `devices` empty — hit this endpoint to confirm your Coolify Application's GPU config.

Example `/analyze` response:

```json
{
  "text": "I loved the movie, it was fantastic!",
  "llm": {
    "sentiment": "positive",
    "confidence": 0.98,
    "reasoning": "The phrase expresses strong enthusiasm and approval.",
    "model": "classroom-chat"
  },
  "local": {
    "sentiment": "positive",
    "confidence": 0.95,
    "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "device": "cuda:0"
  },
  "agreement": true
}
```

## Config (environment variables)

Read at runtime; never hardcode. Set locally via `.env` file (git-ignored) or via Coolify's Environment Variables tab for deployed instances.

| Var                 | Default                                                | Meaning                                     |
|---------------------|--------------------------------------------------------|---------------------------------------------|
| `LITELLM_URL`       | `http://ml-capstone.cs.byu.edu:4000/v1`                | Upstream OpenAI-compat endpoint             |
| `LITELLM_API_KEY`   | `sk-noauth`                                            | Bearer token                                |
| `MODEL`             | `classroom-chat`                                       | LiteLLM model alias                         |
| `LOCAL_MODEL_ID`    | `cardiffnlp/twitter-roberta-base-sentiment-latest`     | HF model id for the local classifier        |
| `DEVICE`            | `cuda:0` if available else `cpu`                       | Where to run the local model                |
| `SKIP_LOCAL_MODEL`  | *(unset)*                                              | Set to `1` in tests to skip loading the HF model |

## Local development

Build and run on your Mac; confirms the code works before pushing. Requires BYU VPN for the LLM call to succeed.

```bash
# Copy the example env file and edit if needed
cp .env.example .env

# Build (slow first time — ~3-5 min; downloads torch + the HF model into the image)
docker build -t sentiment-test-app .

# Run — HF model will load on startup (~30s). GPU is preferred; CPU works too.
docker run --rm -p 8000:8000 --env-file .env sentiment-test-app
```

See `.env.example` for every configurable variable, including the commented-out `CUDA_VISIBLE_DEVICES` pinning line (Pattern A from the GPU-sharing section below).

In another shell:

```bash
curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/health | python3 -m json.tool
curl -sS -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"I loved the movie, it was fantastic!"}' | python3 -m json.tool
```

## GPU access in Coolify

For the local HF classifier to actually use a GPU (not CPU fallback), Coolify needs to hand a GPU to the container. In the Application config:

1. **Enable GPU access** — Coolify's per-app "GPU / Devices" section (or configure a `docker-compose.yml` with the nvidia reservation block)
2. The container needs the NVIDIA runtime — rigel already has `nvidia-container-toolkit` installed
3. Optionally set `DEVICE=cuda:0` in Environment Variables (or let the app auto-pick — see next section)

If you skip the GPU config, the app still works but runs the local model on CPU (~10x slower). Useful to know: the LLM path is entirely unaffected — it always runs on the classroom cluster's GPUs regardless of your container's GPU access.

## Sharing GPUs across many student containers

Rigel has 4× A6000 GPUs; the class can easily have 18+ student containers running at once. Two patterns to spread load:

### Pattern A — per-container GPU pinning (admin-side, recommended)

When Coolify provisions each student Application, set `CUDA_VISIBLE_DEVICES=<n>` where `n = group_number % 4`. Container only sees that one GPU; deterministic, no runtime coordination:

- Group1 → `CUDA_VISIBLE_DEVICES=0`
- Group2 → `CUDA_VISIBLE_DEVICES=1`
- Group3 → `CUDA_VISIBLE_DEVICES=2`
- Group4 → `CUDA_VISIBLE_DEVICES=3`
- Group5 → wraps back to `CUDA_VISIBLE_DEVICES=0`

Handled by the Coolify provisioning script (Phase 19). Students don't need to know about it — inside their container, the assigned GPU shows up as `cuda:0`.

### Pattern B — least-loaded auto-pick (app-side fallback)

If `DEVICE` is unset AND multiple GPUs are visible, the app picks the GPU with the most free VRAM at startup time. See `_detect_device()` in `main.py`. Works when Pattern A isn't in place, or as a defensive default.

**Failure mode:** if many containers restart simultaneously, they all query GPU state at the same instant and pick the same "least loaded" one — a race. Pattern A avoids this. Consider Pattern A the primary strategy; Pattern B the fallback.

### Verifying which GPU your container is using

Hit `/gpu` on your deployed app:

```json
{
  "device_setting": "cuda:0",
  "using_gpu": true,
  "devices": [
    {"index": 0, "name": "NVIDIA RTX A6000", "memory_total_gb": 48.0,
     "memory_free_gb": 47.5, "memory_allocated_gb": 0.5}
  ]
}
```

If `device_count > 1`, your container sees multiple GPUs (Pattern A wasn't applied) and the app is using Pattern B.

## Testing

```bash
pip install -r requirements.txt
pytest tests/ -v
```

Unit tests cover input validation, response shape, model-agreement logic, and the `_extract_json` helper. They set `SKIP_LOCAL_MODEL=1` so the HF pipeline doesn't actually load during tests (would need ~500MB download + GPU). Real integration is checked at deploy time via `/health`.

## CI/CD pipeline

`.github/workflows/ci.yml` implements a **test → deploy-staging → deploy-prod** pipeline:

- **`test`** — unit tests + `docker build`. Runs on every push and every PR. GitHub-hosted runner.
- **`deploy-staging`** — `curl` to Coolify's staging Deploy Webhook. Runs on push to `staging` branch.
- **`deploy-prod`** — `curl` to Coolify's prod Deploy Webhook. Runs on push to `main`.

**Required repo secrets:**
- `COOLIFY_DEPLOY_WEBHOOK_STAGING` — staging Application's Deploy Webhook URL
- `COOLIFY_DEPLOY_WEBHOOK_PROD` — production Application's Deploy Webhook URL

**Coolify-side config so tests gate deploys:**
- Turn OFF Coolify's own "Auto Deploy" toggle so pushes only deploy via Actions
- Point Coolify's health check at `/health` (Coolify default) so failing LLM or local-model paths fail the deploy
- Both Applications set `LITELLM_URL`, `MODEL`, and (if GPU is available) `DEVICE=cuda:0`

## Branch flow

- Feature branch → PR into `staging` → merge → auto-deploys to `<group>-staging.ml-capstone.cs.byu.edu`
- Manual QA on staging URL
- PR from `staging` into `main` → merge → auto-deploys to `<group>.ml-capstone.cs.byu.edu`
- Rollback: `git revert <bad-sha> && git push origin main`

## Cluster smoke test integration

The top-level `smoke-test-cluster.sh` exercises this app:

```bash
./smoke-test-cluster.sh -p 8100                 # hit the Coolify-deployed instance at rigel:8100
SENTIMENT_URL=http://Group1.ml-capstone.cs.byu.edu ./smoke-test-cluster.sh
```
