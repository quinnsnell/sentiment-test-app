> **Starting point for BYU CS ml-capstone students:** this is the **reference / growth target**, not the starting template. Students first template [`byu-ml-capstone/hello-world-app`](https://github.com/byu-ml-capstone/hello-world-app) and grow it incrementally into something structurally like this repo, following `student-guide.md` Section 1. The final result should look a lot like what's here.
>
> If you jumped straight to this repo without going through the guide, you'll be missing context on the Coolify deploy pipeline, the `byu-ml-capstone-coolify` GitHub App, the roster-based Coolify Team provisioning, and how to test integration with the classroom LiteLLM. Start with the guide instead: `github.com/quinnsnell/ml-capstone-platform`.

---

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

## Code layout

Repo root holds orchestration (compose files, CI workflow, scripts). All app code lives in the `sentiment/` subdirectory — each service = one subdirectory, same pattern as [`byu-ml-capstone/hello-world-app`](https://github.com/byu-ml-capstone/hello-world-app).

```
sentiment-test-app/
├── docker-compose.yaml           # production compose (Coolify reads this)
├── docker-compose.override.yml   # local-dev only (host port bind); ignored by Coolify
├── smoke-test.sh                 # local or remote endpoint smoke test
├── integration-test.sh           # Tier-3 integration tests against a deployed URL
├── .github/workflows/            # CI: test job + Coolify deploy webhooks + base-image build
├── .env.example                  # config template; copy to .env for local dev
└── sentiment/                    # THE APP
    ├── main.py                   #   FastAPI app + endpoints
    ├── config.py                 #   env vars + APP_VERSION
    ├── device.py                 #   GPU / CPU detection
    ├── schemas.py                #   Pydantic request/response shapes
    ├── llm_client.py             #   classroom LiteLLM client
    ├── local_classifier.py       #   local HF pipeline
    ├── requirements.txt
    ├── Dockerfile                #   thin FROM the pre-built base image
    ├── Dockerfile.base           #   base image with torch + HF model baked in
    ├── .dockerignore
    ├── conftest.py               #   sets SKIP_LOCAL_MODEL=1 for tests
    └── tests/                    #   test_api.py, test_extract_json.py
```

Business logic split across small focused modules; `main.py` stays thin.

| File | Responsibility |
|---|---|
| `sentiment/main.py` | FastAPI app, endpoints (`/ready`, `/gpu`, `/health`, `/analyze`), startup lifespan |
| `sentiment/config.py` | Environment-variable reads + `APP_VERSION` |
| `sentiment/device.py` | GPU / CPU detection (`detect_device`, `DEVICE`) + `/gpu` status |
| `sentiment/schemas.py` | Pydantic shapes: `AnalyzeRequest`, `LLMResult`, `LocalResult`, `AnalyzeResponse` |
| `sentiment/llm_client.py` | LiteLLM: `classify_llm`, `extract_json`, `health_check` |
| `sentiment/local_classifier.py` | HF pipeline: `load_pipeline`, `classify_local`, `health_check` |
| `sentiment/tests/` | `test_api.py` (HTTP surface), `test_extract_json.py` (parser) |

The **why** — separation of concerns:

- `main.py` doesn't know how to talk to LiteLLM or how a HuggingFace pipeline works. It just wires HTTP verbs to functions in other modules. If you swap FastAPI for another web framework, only `main.py` changes.
- `llm_client.py` doesn't know anything about HTTP endpoints. It's callable from a script, a Jupyter notebook, or a batch job.
- `device.py` doesn't know about sentiment or FastAPI. It could be reused verbatim in any GPU-using app.
- Tests can patch specific modules without touching everything else. `test_api.py` patches `main.llm_client.classify_llm` — it doesn't need to know what HTTP calls that function makes.

This is a real pattern you'll see in production Python apps. The rule of thumb is **each file does one thing well**, and the file names announce what that thing is. When a file grows past ~150 lines or starts having "and" in its purpose statement, it usually wants to be split.

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

The recommended loop is:

1. **Edit code.**
2. **`./smoke-test.sh`** — brings up compose, waits for `/ready`, hits every endpoint. Fails fast without a 5–6 min Coolify roundtrip.
3. Push only when green.

### Quick start

```bash
cp .env.example .env                  # local config, git-ignored
./smoke-test.sh                       # local: builds + starts + endpoint checks
```

The script:

- Runs `docker compose up -d --build` (compose reads `.env` from the repo root automatically)
- Polls `/ready` (up to 180s while the HF model loads)
- Hits `/ready`, `/gpu`, `/health`, `/analyze` (positive + negative samples)
- Leaves the container running so you can keep poking

Stop with `docker compose down` when done. Requires BYU VPN for `/health` and `/analyze` (they call the classroom LiteLLM).

**Also works against a deployed URL** — pass it as an argument to skip the local docker steps:

```bash
./smoke-test.sh http://sentiment-test-app-staging.ml-capstone.cs.byu.edu
```

Useful right after a Coolify deploy: same checks, remote target. What passes locally should pass remotely; a divergence points at a Coolify-only bug (env vars, GPU allocation, networking).

**For fast iteration on unit tests only** (skip Docker entirely, run pytest directly):

```bash
cd sentiment
pip install -r requirements.txt httpx pytest
SKIP_LOCAL_MODEL=1 pytest -v
```

### Manual smoke tests

If you'd rather drive it by hand:

```bash
export SERVICE_FQDN_SENTIMENT=http://localhost:8000     # stubs the compose interpolation
docker compose up -d --build

curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/gpu | python3 -m json.tool
curl -sS http://127.0.0.1:8000/health | python3 -m json.tool
curl -sS -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"I loved the movie!"}' | python3 -m json.tool

docker compose down                                     # when done
```

See `.env.example` for every configurable variable, including the commented-out `CUDA_VISIBLE_DEVICES` pinning line (Pattern A from the GPU-sharing section below).

### The full "update-test-PR-deploy" loop

The intended development flow for the whole class:

1. Branch off `staging`: `git checkout staging && git checkout -b your-feature`
2. Edit code
3. **`./smoke-test.sh`** — green before you push
4. `git push origin your-feature`
5. Open PR into `staging` — GitHub Actions `test` job runs on the PR
6. Merge the PR — pushes to `staging`, triggers `deploy-staging`
7. Check staging URL manually
8. Open PR from `staging` → `main` when staging looks good
9. Merge — pushes to `main`, triggers `deploy-prod`

`smoke-test.sh` gates step 4 (before push), Actions gates steps 6 and 9 (before deploy). Two safety layers.

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
cd sentiment
pip install -r requirements.txt httpx pytest
pytest tests/ -v
```

Unit tests cover input validation, response shape, model-agreement logic, and the `_extract_json` helper. They set `SKIP_LOCAL_MODEL=1` via `sentiment/conftest.py` so the HF pipeline doesn't actually load during tests (would need ~500MB download + GPU). Real integration is checked at deploy time via `/health`.

## CI/CD pipeline

`.github/workflows/ci.yml` implements a **test → deploy-staging → deploy-prod** pipeline:

- **`test`** — runs pytest with lightweight deps (no `docker build`, no torch install — see "How this app is packaged" below for why). Runs on every push and every PR. GitHub-hosted runner. ~15s.
- **`deploy-staging`** — `curl` to Coolify's staging Deploy Webhook. Runs on push to `staging` branch.
- **`deploy-prod`** — `curl` to Coolify's prod Deploy Webhook. Runs on push to `main`.

**Required repo secrets:**
- `COOLIFY_DEPLOY_WEBHOOK_STAGING` — staging Application's Deploy Webhook URL
- `COOLIFY_DEPLOY_WEBHOOK_PROD` — production Application's Deploy Webhook URL

**Coolify-side config so tests gate deploys:**
- Turn OFF Coolify's own "Auto Deploy" toggle so pushes only deploy via Actions
- Point Coolify's health check at `/health` (Coolify default) so failing LLM or local-model paths fail the deploy
- Both Applications set `LITELLM_URL`, `MODEL`, and (if GPU is available) `DEVICE=cuda:0`

## How this app is packaged (and why)

This section is a teaching detour. If you just want to ship code, you can skip it — but it explains a real production pattern you'll encounter at any company that ships ML apps, and it's why your deploys are ~30 seconds instead of ~5 minutes.

### The problem

The container image for this app contains:

- Python 3.12 (~50 MB)
- `torch` and CUDA runtime libraries (~750 MB)
- `transformers` and friends (~500 MB)
- The pre-downloaded HuggingFace sentiment model (~500 MB)
- Your actual code — `main.py` — (about 8 KB)

That's **~1.8 GB of heavy dependencies vs. 8 KB of code**. Almost all pushes only change the 8 KB. If you build the image from scratch on every push, you pay the ~5 minutes to reinstall torch and re-download the model **every time** — even when nothing about them changed.

### The solution — two Dockerfiles

This repo has two Docker files instead of one:

```
Dockerfile.base   ← heavy stuff: Python + torch + transformers + HF model
Dockerfile        ← thin layer: FROM the base image + COPY main.py
```

**`Dockerfile.base`** — rebuilt only when `requirements.txt` or `Dockerfile.base` itself changes. When rebuilt, the resulting image is pushed to GitHub Container Registry (GHCR) at `ghcr.io/quinnsnell/sentiment-test-app-base:latest`. This is our long-lived, cached "everything except my code" image.

**`Dockerfile`** — starts with `FROM ghcr.io/quinnsnell/sentiment-test-app-base:latest` and just adds `main.py`. Rebuilds on every push. Since almost nothing new happens in the build (base is already cached, only one tiny COPY + tag), it's seconds.

### What triggers each rebuild

Two GitHub Actions workflows handle this split:

| Workflow file | Triggers on | Time | Output |
|---|---|---|---|
| `.github/workflows/build-base.yml` | Push touching `sentiment/requirements.txt` or `sentiment/Dockerfile.base` | ~5-8 min (first) / ~1-2 min (cached) | Pushes base image to GHCR |
| `.github/workflows/ci.yml` | Push touching anything except docs | ~15s test + ~30-60s Coolify build | Runs tests, triggers Coolify deploy |

The base workflow uses `docker/build-push-action` with a **BuildKit registry cache** (`cache-to: ghcr.io/...:buildcache`). Second and later runs pull unchanged layers from GHCR — even inside a fresh Actions runner, only the layers that actually changed get rebuilt.

### Where the caches actually live — the crucial insight

Docker layer caching is only useful if there's a persistent place to store the cache between runs. Different parts of a CI/CD pipeline have wildly different cache behavior:

| Where docker runs | Cache persistence | Effect |
|---|---|---|
| **GitHub Actions `test` job** | We don't do docker there at all | N/A |
| **GitHub Actions `build-base` job** | Fresh VM each run → no local cache | Compensated by BuildKit registry cache to GHCR |
| **GitHub Actions inside runners in general** | Ephemeral | Every image pull is fresh; every build starts from zero unless you configure external cache |
| **Coolify's build container on rigel** | Persistent local docker cache | Pulls base image once, reuses forever until the tag moves |
| **Your Mac when you run `smoke-test.sh`** | Persistent local docker cache | Same as Coolify — cached after first pull |
| **Docker registries (GHCR, Docker Hub)** | Persistent by nature | Can act as a distributed cache via `cache-from`/`cache-to` |

The split-image pattern **only pays off where caches persist**. That's Coolify (our main win) and your Mac (your dev loop). It doesn't help GitHub-hosted runners much, which is why we don't do `docker build` in the `test` job at all — we just run `pytest` with lightweight deps installed via pip.

### The performance numbers

Measured on a code-only push (main.py version bump):

| Step | Before split | After split, first push | After split, subsequent pushes |
|---|---|---|---|
| Actions `test` job | ~2 min | ~15s | ~15s |
| Coolify build | ~5 min | ~5 min (fresh base pull) | ~5-10s |
| Container start + model load | ~30-60s | ~30-60s | ~30-60s |
| Health check grace | ~30s | ~30s | ~30s |
| **Total push-to-live** | **~7 min** | **~6 min** | **~1 min** |

That's a 7x improvement on the common case (code changes). Requirements changes still incur the ~5-8 min base rebuild, but they're rare — you're not adding a new pip package every push.

### Trade-offs

**Costs:**
- Two Dockerfiles to keep in sync (`Dockerfile.base` and `Dockerfile`)
- Extra Actions workflow to run + maintain
- The base image is publicly readable in GHCR (fine here — no secrets — but something to think about in industry)
- First-ever pull on any host is still slow (~2 GB over the network)
- Coordination: if `requirements.txt` changes but the base workflow hasn't finished yet when Coolify tries to deploy, the app will pull the OLD base image

**When it's worth it:**
- Heavy dependencies (>500 MB total)
- Dependencies change less often than app code (typical for ML apps)
- You care about deploy latency (production apps, developer flow)

**When it's overkill:**
- Small apps (< 200 MB total)
- Dependencies churn as fast as code
- Deploy latency doesn't matter (batch jobs, long-running services)
- You have exactly one deployment target and rebuilds are rare

### Alternative patterns you'll see in production

For context, the industry has other ways to solve the "heavy dependencies, fast deploys" problem:

**1. Model-as-a-service (the classroom's `classroom-chat` pattern).** Don't ship the model in the app at all. Run it as its own service (vLLM, Triton, TorchServe) that many apps call via API. This is what LiteLLM does for us — Qwen3-Coder-Next lives on castor+pollux, students' apps are tiny clients. **App builds go from ~5 min to seconds.** Trade-off: you lose "app owns model" isolation; you have another service to run.

**2. Persistent volume mounted at runtime.** Don't bake the model into the image at all; mount it from a persistent volume when the container starts. `main.py` still calls `AutoModel.from_pretrained('/models/roberta')`. Image is tiny, first container start is slow (loading model from disk into VRAM), subsequent starts are fast. Trade-off: infrastructure complexity (need a persistent volume system like Kubernetes PV or NFS).

**3. Straight single-Dockerfile bake with layer caching.** Just one `Dockerfile` that installs everything, but rely on Docker's own layer caching to skip unchanged steps. Works fine when your build host has a stable cache. Falls over on ephemeral CI runners — which is why our first Coolify builds were so slow.

**4. Sidecar model service.** In Kubernetes, deploy a small "model sidecar" alongside the app container. App calls localhost:8500, sidecar has the model loaded. Similar to model-as-a-service but co-located. Common in Google internal setups.

The right pattern depends on your scale and constraints. For a classroom demo where the goal is to show you can use a local GPU model, the split-image approach is a good middle ground: educational, still-real production pattern, works on rigel today. In a real job you might pick #1 or #2 instead.

### The takeaway

Fast deploys don't come from "make Docker faster" — they come from **structuring your image so the parts that change often are separated from the parts that rarely change**, then **making sure the cache lives where builds happen**. That's the whole trick.

## Branch flow

- Feature branch → PR into `staging` → merge → auto-deploys to `<your-repo>-staging.ml-capstone.cs.byu.edu`
- Manual QA on staging URL
- PR from `staging` into `main` → merge → auto-deploys to `<your-repo>.ml-capstone.cs.byu.edu`
- Rollback: `git revert <bad-sha> && git push origin main`

## Cluster smoke test integration

The top-level `smoke-test-cluster.sh` exercises this app:

```bash
./smoke-test-cluster.sh -p 8100                 # hit the Coolify-deployed instance at rigel:8100
SENTIMENT_URL=http://Group1.ml-capstone.cs.byu.edu ./smoke-test-cluster.sh
```
