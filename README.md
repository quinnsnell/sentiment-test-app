# sentiment-test-app

Tiny FastAPI service that proxies sentiment analysis through the classroom LiteLLM. Used as the Coolify-deploy smoke test.

## What it does

- `POST /analyze { "text": "..." }` → asks `classroom-chat` via LiteLLM for a sentiment classification, returns strict JSON.
- `GET /health` → self-report + which LiteLLM upstream it's pointed at.

## Config (env vars)

| Var               | Default                              | Meaning                          |
|-------------------|--------------------------------------|----------------------------------|
| `LITELLM_URL`     | `http://rigel.cs.byu.edu:4000/v1`    | Upstream OpenAI-compat endpoint  |
| `LITELLM_API_KEY` | `sk-noauth`                          | Bearer token                     |
| `MODEL`           | `classroom-chat`                     | Model alias to hit               |

## Deploy paths (pick one)

### 1. Local sanity check (fastest)

Build and run on your Mac; confirms the code works before touching Coolify. Uses your Mac's Docker daemon and hits LiteLLM on rigel over the VPN.

```bash
cd sentiment-test-app
docker build -t sentiment-test .
docker run --rm -p 8000:8000 -e LITELLM_URL=http://rigel.cs.byu.edu:4000/v1 sentiment-test
```

Then in another shell:

```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"I loved the movie, it was fantastic!"}' | python3 -m json.tool
```

### 2. Coolify — Public repository (no GitHub App needed)

Once you have a public repo containing this directory:

1. Coolify UI → **New Resource → Application → Public Repository**.
2. Paste the repo URL, pick the `sentiment-test-app/` subdirectory (or move these files to the repo root).
3. Build pack: **Dockerfile**.
4. Env: `LITELLM_URL=http://rigel.cs.byu.edu:4000/v1` (Docker network on rigel — LiteLLM is reachable at `172.17.0.1:4000` if you'd rather stay on the docker bridge).
5. Deploy. Note the auto-generated hostname (e.g. `sentiment-test.apps.class.byu.edu`).

### 3. Coolify — Docker Image (needs a public registry account)

```bash
docker build -t ghcr.io/<you>/sentiment-test:latest .
docker push ghcr.io/<you>/sentiment-test:latest
```

Coolify UI → **New Resource → Application → Docker Image** → paste the image name → deploy.

### 4. Coolify — GitHub App (needs Phase 13 + 14 done)

Once the class GitHub App is wired: **New Resource → Application → Private Repository (via GitHub App)**. Auto-redeploy on push works from then on.

## Smoke-testing after deploy

Once the app is deployed and reachable at some URL, run the cluster smoke test with that URL exported:

```bash
SENTIMENT_URL=https://sentiment-test.apps.class.byu.edu ./smoke-test-cluster.sh
```

It will exercise `/health` and a `POST /analyze` and print the classification.
