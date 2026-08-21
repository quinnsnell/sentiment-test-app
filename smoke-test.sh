#!/usr/bin/env bash
# =============================================================================
# smoke-test.sh — hit every important endpoint and report what came back.
#
# Two modes:
#
#   Local (no argument):
#     ./smoke-test.sh
#     -> docker compose up -d --build, wait for /ready (model loads
#        into GPU/CPU — up to 180s), curl the endpoints against
#        localhost:8000, leave the container running.
#
#   Remote (base URL argument):
#     ./smoke-test.sh http://your-app.ml-capstone.cs.byu.edu
#     -> skips docker compose entirely; curls the endpoints against
#        the given URL. Useful for smoke-testing a Coolify deploy
#        (staging or prod) from your laptop after a push.
#
# REQUIRES BYU VPN — the /health and /analyze checks call the classroom
# LiteLLM, which is VPN-only.
#
# Both modes hit /ready /gpu /health /analyze (positive + negative).
# =============================================================================
set -uo pipefail

# ---- Args ------------------------------------------------------------------
if [ $# -eq 0 ]; then
    MODE=local
    BASE_URL="http://localhost:8000"
else
    MODE=remote
    BASE_URL="${1%/}"    # strip trailing slash
fi

# ---- Colors ---------------------------------------------------------------
if [[ -t 1 ]]; then
    G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'
    D=$'\033[90m'; B=$'\033[1m'; Z=$'\033[0m'
else
    G=""; R=""; Y=""; D=""; B=""; Z=""
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

# ---- Local: build + start the compose stack -------------------------------

if [ "$MODE" = "local" ]; then
    # Stub the ${SERVICE_FQDN_SENTIMENT} interpolation in docker-compose.yaml
    # so compose doesn't warn about an undefined variable.
    export SERVICE_FQDN_SENTIMENT="$BASE_URL"

    printf "\n%ssmoke-test.sh%s   local mode\n" "$B$Y" "$Z"
    hr
    printf "\n%sBuilding + starting (docker compose)%s\n" "$B" "$Z"
    printf "%s(first build ~5 min for torch+HF; subsequent much faster)%s\n" "$D" "$Z"

    docker compose down --remove-orphans >/dev/null 2>&1 || true
    if docker compose up -d --build ; then
        record "docker compose up" PASS
    else
        record "docker compose up" FAIL
        printf "\n${R}Build/start failed — stopping here.${Z}\n"
        exit 1
    fi
else
    printf "\n%ssmoke-test.sh%s   remote mode: %s\n" "$B$Y" "$Z" "$BASE_URL"
    hr
fi

# ---- Wait for /ready ------------------------------------------------------

printf "\n%sWait for /ready%s (up to 180s while the model loads)\n" "$B" "$Z"
elapsed=0
until curl -sSf -m 3 "$BASE_URL/ready" >/dev/null 2>&1; do
    if (( elapsed >= 180 )); then
        printf "  ${R}FAIL${Z}  /ready never responded within 180s\n"
        if [ "$MODE" = "local" ]; then
            docker compose logs sentiment 2>&1 | tail -30 | sed "s|^|        ${D}| ; s|\$|${Z}|"
        fi
        record "wait for /ready" FAIL
        exit 1
    fi
    sleep 3
    elapsed=$((elapsed + 3))
    printf "."
done
printf "\n  ${G}PASS${Z}  /ready responded after ~%ds\n" "$elapsed"
record "wait for /ready" PASS

# ---- Endpoint checks ------------------------------------------------------

printf "\n%sEndpoint checks%s\n" "$B" "$Z"

check "/ready returns ready:true" bash -c "\
    body=\$(curl -sSf -m 5 '$BASE_URL/ready') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d[\"ready\"] is True'"

check "/gpu returns valid structure" bash -c "\
    body=\$(curl -sSf -m 5 '$BASE_URL/gpu') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert \"device_setting\" in d and \"using_gpu\" in d and \"devices\" in d'"

# Informational snapshot of /gpu
printf "  ${D}/gpu snapshot:${Z}\n"
curl -sSf -m 5 "$BASE_URL/gpu" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f\"    device:  {d.get('device_setting')}\")
print(f\"    using_gpu: {d.get('using_gpu')}\")
print(f\"    device_count: {d.get('device_count')}\")
" 2>/dev/null || true

check "/health returns 200 with both models ok (requires VPN)" bash -c "\
    body=\$(curl -sSf -m 30 '$BASE_URL/health') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert d[\"ok\"] and d[\"llm\"][\"ok\"] and d[\"local\"][\"ok\"]'"

check "/analyze rejects missing text (422)" bash -c "\
    code=\$(curl -sS -o /dev/null -w '%{http_code}' -X POST '$BASE_URL/analyze' \
      -H 'Content-Type: application/json' -d '{}') && \
    [[ \"\$code\" == '422' ]]"

check "/analyze positive sample" bash -c "\
    body=\$(curl -sSf -m 30 -X POST '$BASE_URL/analyze' \
      -H 'Content-Type: application/json' \
      -d '{\"text\":\"I loved the movie!\"}') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert d[\"llm\"][\"sentiment\"] == \"positive\", d; \
        assert d[\"local\"][\"sentiment\"] == \"positive\", d'"

check "/analyze negative sample" bash -c "\
    body=\$(curl -sSf -m 30 -X POST '$BASE_URL/analyze' \
      -H 'Content-Type: application/json' \
      -d '{\"text\":\"terrible service, disappointing food\"}') && \
    echo \"\$body\" | python3 -c 'import json,sys; d=json.load(sys.stdin); \
        assert d[\"llm\"][\"sentiment\"] == \"negative\", d; \
        assert d[\"local\"][\"sentiment\"] == \"negative\", d'"

# ---- Summary --------------------------------------------------------------

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
    printf "  - Docker daemon not running (local mode)\n"
    printf "  - Deploy hasn't finished yet (remote mode) — check Coolify Deployments tab\n"
    printf "  - Build cache stale (local mode) — try 'docker compose down -v' + rerun\n"
    exit 1
fi

printf "${G}Green across the board.${Z}"
if [ "$MODE" = "local" ]; then
    printf " Container still running on http://localhost:8000\n"
    printf "  logs:  docker compose logs -f\n"
    printf "  stop:  docker compose down\n"
else
    printf " Remote deploy looks healthy: %s\n" "$BASE_URL"
fi
exit 0
