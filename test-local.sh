#!/usr/bin/env bash
# =============================================================================
# test-local.sh — full local test loop for sentiment-test-app.
#
# Run this before opening a PR. Catches most failures in <30 seconds without
# a 5-6 minute Coolify roundtrip.
#
# Pipeline:
#   1. Unit tests (pytest, ~0.5s)
#   2. Docker build .
#   3. Start the container in the background
#   4. Wait for /ready
#   5. Hit each endpoint (/ready, /gpu, /health, /analyze)
#   6. Clean up the container
#
# Requires: docker, python3, curl. VPN required for the /health and /analyze
# checks to succeed (they call the classroom LiteLLM).
#
# Options:
#   --skip-unit         skip pytest step
#   --skip-build        skip docker build (reuse existing image)
#   --skip-live         skip the container-run + endpoint checks (unit tests only)
#   --port N            host port to publish (default 8001)
#   --keep-running      leave the container running at the end for manual poking
# =============================================================================
set -uo pipefail

# ---- Defaults --------------------------------------------------------------
PORT=8001
DO_UNIT=1
DO_BUILD=1
DO_LIVE=1
KEEP_RUNNING=0
CONTAINER="sentiment-test-app-local-$$"
IMAGE="sentiment-test-app:local"

# ---- Args ------------------------------------------------------------------
while (( "$#" )); do
    case "$1" in
        --skip-unit)      DO_UNIT=0;    shift ;;
        --skip-build)     DO_BUILD=0;   shift ;;
        --skip-live)      DO_LIVE=0;    shift ;;
        --keep-running)   KEEP_RUNNING=1; shift ;;
        --port)           PORT="$2";    shift 2 ;;
        -h|--help)
            sed -n '2,/^# ===*$/{ /^# ===*$/d; s/^# \{0,1\}//p; }' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2 ;;
    esac
done

# ---- Colors ---------------------------------------------------------------
if [[ -t 1 ]]; then
    G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; C=$'\033[36m'
    D=$'\033[90m'; B=$'\033[1m'; Z=$'\033[0m'
else
    G=""; R=""; Y=""; C=""; D=""; B=""; Z=""
fi

PASS=0
FAIL=0
FAILED=()

record() {
    local name="$1" status="$2"
    if [[ "$status" == PASS ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED+=("$name")
    fi
}

check() {
    # check "human name" cmd args...
    local name="$1"; shift
    local out ec
    out=$("$@" 2>&1); ec=$?
    if (( ec == 0 )); then
        printf "  ${G}PASS${Z}  %s\n" "$name"
        record "$name" PASS
    else
        printf "  ${R}FAIL${Z}  %s\n" "$name"
        printf "%s\n" "$out" | head -5 | sed "s|^|        ${D}| ; s|\$|${Z}|"
        record "$name" FAIL
    fi
}

hr() { printf "%s%s%s\n" "$D" "$(printf '─%.0s' {1..70})" "$Z"; }

# ---- Cleanup -------------------------------------------------------------
cleanup() {
    if (( KEEP_RUNNING )) && [[ $DO_LIVE -eq 1 ]]; then
        printf "\n%sContainer left running at http://127.0.0.1:%s%s\n" "$Y" "$PORT" "$Z"
        printf "%sStop it with: docker stop %s%s\n" "$D" "$CONTAINER" "$Z"
        return
    fi
    if docker ps -aq --filter "name=^${CONTAINER}$" | grep -q .; then
        docker stop "$CONTAINER" >/dev/null 2>&1 || true
        docker rm   "$CONTAINER" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# ==========================================================================
# 1. Unit tests
# ==========================================================================

printf "\n%stest-local.sh%s   port=%d\n" "$B$Y" "$Z" "$PORT"
hr

if (( DO_UNIT )); then
    printf "\n%sUnit tests (pytest)%s\n" "$B" "$Z"
    if command -v pytest >/dev/null; then
        pytest -q tests/ && record "pytest" PASS || record "pytest" FAIL
    elif command -v python3 >/dev/null; then
        python3 -m pytest -q tests/ && record "pytest (python3 -m)" PASS || record "pytest (python3 -m)" FAIL
    else
        printf "  ${R}FAIL${Z}  pytest not found (install with: pip install pytest)\n"
        record "pytest" FAIL
    fi
fi

if (( ! DO_LIVE )); then
    hr
    printf "%d checks  ${G}%d passed${Z}  ${R}%d failed${Z}\n\n" "$((PASS+FAIL))" "$PASS" "$FAIL"
    exit $(( FAIL > 0 ))
fi

# ==========================================================================
# 2. Docker build
# ==========================================================================

if (( DO_BUILD )); then
    printf "\n%sDocker build%s   (first build ~5 min for torch+HF; subsequent much faster)\n" "$B" "$Z"
    if docker build -t "$IMAGE" . ; then
        record "docker build" PASS
    else
        record "docker build" FAIL
        printf "\n${R}Build failed — stopping here.${Z}\n"
        exit 1
    fi
fi

# ==========================================================================
# 3. Run container
# ==========================================================================

printf "\n%sStart container%s\n" "$B" "$Z"

# Assemble env from .env if present
ENV_FLAGS=()
if [[ -f .env ]]; then
    ENV_FLAGS+=(--env-file .env)
fi

# Try with GPU access first (--gpus all). If that fails because no NVIDIA
# runtime is installed on this machine (typical Mac), fall back to CPU.
if docker run -d --rm --name "$CONTAINER" -p "${PORT}:8000" \
      --gpus all "${ENV_FLAGS[@]}" "$IMAGE" >/dev/null 2>&1; then
    printf "  ${G}PASS${Z}  container started with --gpus all\n"
    record "container start" PASS
elif docker run -d --rm --name "$CONTAINER" -p "${PORT}:8000" \
        "${ENV_FLAGS[@]}" "$IMAGE" >/dev/null 2>&1; then
    printf "  ${G}PASS${Z}  container started (no GPU on this host; CPU fallback)\n"
    record "container start" PASS
else
    printf "  ${R}FAIL${Z}  could not start container\n"
    docker logs "$CONTAINER" 2>&1 | tail -20 | sed "s|^|        ${D}| ; s|\$|${Z}|" || true
    record "container start" FAIL
    exit 1
fi

# ==========================================================================
# 4. Wait for /ready — the container needs ~30-60s to load the HF model
# ==========================================================================

printf "\n%sWait for /ready%s (up to 180s while model loads)\n" "$B" "$Z"
READY_URL="http://127.0.0.1:${PORT}/ready"
elapsed=0
until curl -sSf -m 3 "$READY_URL" >/dev/null 2>&1; do
    if (( elapsed >= 180 )); then
        printf "  ${R}FAIL${Z}  /ready never responded within 180s\n"
        docker logs "$CONTAINER" 2>&1 | tail -30 | sed "s|^|        ${D}| ; s|\$|${Z}|"
        record "wait for /ready" FAIL
        exit 1
    fi
    sleep 3
    elapsed=$((elapsed + 3))
    printf "."
done
printf "\n  ${G}PASS${Z}  /ready responded after ~%ds\n" "$elapsed"
record "wait for /ready" PASS

# ==========================================================================
# 5. Endpoint checks
# ==========================================================================

BASE="http://127.0.0.1:${PORT}"

printf "\n%sEndpoint checks%s\n" "$B" "$Z"

# /ready
check "/ready returns ready:true" bash -c "\
    body=\$(curl -sSf -m 5 '$BASE/ready') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"ready\"] is True'"

# /gpu (structural)
check "/gpu returns valid structure" bash -c "\
    body=\$(curl -sSf -m 5 '$BASE/gpu') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert \"device_setting\" in d and \"using_gpu\" in d and \"devices\" in d'"

# Show what /gpu says (informational)
printf "  ${D}/gpu snapshot:${Z}\n"
curl -sSf -m 5 "$BASE/gpu" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"    device:  {d.get('device_setting')}\")
print(f\"    using_gpu: {d.get('using_gpu')}\")
print(f\"    device_count: {d.get('device_count')}\")
"

# /health (deep — requires VPN)
check "/health returns 200 with both models ok (requires VPN)" bash -c "\
    body=\$(curl -sSf -m 30 '$BASE/health') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert d[\"ok\"] and d[\"llm\"][\"ok\"] and d[\"local\"][\"ok\"]'"

# /analyze validation
check "/analyze rejects missing text (422)" bash -c "\
    code=\$(curl -sS -o /dev/null -w '%{http_code}' -X POST '$BASE/analyze' \
      -H 'Content-Type: application/json' -d '{}') && \
    [[ \"\$code\" == '422' ]]"

# /analyze positive
check "/analyze positive sample" bash -c "\
    body=\$(curl -sSf -m 30 -X POST '$BASE/analyze' \
      -H 'Content-Type: application/json' \
      -d '{\"text\":\"I loved the movie!\"}') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert d[\"llm\"][\"sentiment\"] == \"positive\", d; \
        assert d[\"local\"][\"sentiment\"] == \"positive\", d'"

# /analyze negative
check "/analyze negative sample" bash -c "\
    body=\$(curl -sSf -m 30 -X POST '$BASE/analyze' \
      -H 'Content-Type: application/json' \
      -d '{\"text\":\"terrible service, disappointing food\"}') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert d[\"llm\"][\"sentiment\"] == \"negative\", d; \
        assert d[\"local\"][\"sentiment\"] == \"negative\", d'"

# ==========================================================================
# Summary
# ==========================================================================

TOTAL=$((PASS + FAIL))
printf "\n"; hr
printf "%d checks  ${G}%d passed${Z}  ${R}%d failed${Z}\n\n" "$TOTAL" "$PASS" "$FAIL"

if (( FAIL > 0 )); then
    printf "${R}Failed:${Z}\n"
    for name in "${FAILED[@]}"; do
        printf "  - %s\n" "$name"
    done
    printf "\n${Y}Common causes:${Z}\n"
    printf "  - Not on BYU VPN (blocks /health and /analyze — they call the classroom LLM)\n"
    printf "  - Docker daemon not running\n"
    printf "  - Something else on port %s (try --port %s)\n" "$PORT" "$((PORT + 1))"
    printf "  - Build cache stale — rerun with a clean image or 'docker system prune'\n"
    exit 1
fi

printf "${G}Green across the board.${Z} Safe to push.\n"
exit 0
