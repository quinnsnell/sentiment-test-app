#!/usr/bin/env bash
# =============================================================================
# integration-test.sh — real integration tests against a live deployed instance.
#
# REQUIRES BYU VPN — target URLs are on ml-capstone.cs.byu.edu (VPN-only).
#
# Contrast with the other testing tools in this repo:
#
#   test-local.sh       Builds the container locally, runs unit tests +
#                       endpoint checks against the local container. Run
#                       BEFORE you push.
#
#   Coolify /health     Deep health check invoked by Coolify at deploy time.
#                       Verifies the deployed container can reach real
#                       dependencies (LLM + local model). Deploy fails if
#                       /health returns non-2xx.
#
#   integration-test.sh THIS SCRIPT. Hits a live deployed URL with real
#                       data — happy paths, edge cases, response-shape
#                       contracts, optional stress. Run AFTER staging
#                       deploy, BEFORE opening the PR from staging → main.
#
# Usage:
#   ./integration-test.sh                     # defaults to the sentiment-test-app URL
#   ./integration-test.sh --url <URL>         # any deployed instance
#   ./integration-test.sh --staging           # replace prod hostname with -staging
#   ./integration-test.sh --stress            # add concurrency/load checks
#   ./integration-test.sh --json <file>       # write per-check results to a JSON file
# =============================================================================
set -uo pipefail

# ---- Defaults ------------------------------------------------------------
: "${TARGET_URL:=http://sentiment-test-app.ml-capstone.cs.byu.edu}"
INCLUDE_STRESS=0
JSON_OUT=""

# ---- Arg parsing ---------------------------------------------------------
while (( "$#" )); do
    case "$1" in
        --url)     TARGET_URL="$2"; shift 2 ;;
        --staging)
            # Insert -staging before .ml-capstone.
            TARGET_URL="${TARGET_URL/.ml-capstone.cs.byu.edu/-staging.ml-capstone.cs.byu.edu}"
            shift ;;
        --stress)  INCLUDE_STRESS=1; shift ;;
        --json)    JSON_OUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -uo/{ /^set -uo/d; s/^# \{0,1\}//p; }' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; echo "See --help." >&2; exit 2 ;;
    esac
done

# ---- Colors --------------------------------------------------------------
if [[ -t 1 ]]; then
    G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; C=$'\033[36m'
    D=$'\033[90m'; B=$'\033[1m'; Z=$'\033[0m'
else
    G=""; R=""; Y=""; C=""; D=""; B=""; Z=""
fi

PASS=0
FAIL=0
FAILED=()
# Parallel arrays for JSON export (indexed identically). Easier than
# building JSON strings inline (shell brace expansion vs. python dict syntax
# gets messy).
CHECK_NAMES=()
CHECK_STATUS=()
CHECK_MS=()
CHECK_REASON=()

now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

# ---- Result helpers ------------------------------------------------------
record_pass() {
    local name="$1" ms="$2"
    printf "  ${G}PASS${Z}  %-55s ${D}%dms${Z}\n" "$name" "$ms"
    PASS=$((PASS + 1))
    CHECK_NAMES+=("$name")
    CHECK_STATUS+=("pass")
    CHECK_MS+=("$ms")
    CHECK_REASON+=("")
}

record_fail() {
    local name="$1" ms="$2" reason="$3"
    printf "  ${R}FAIL${Z}  %-55s ${D}%dms${Z}\n" "$name" "$ms"
    printf "        ${D}%s${Z}\n" "$reason"
    FAIL=$((FAIL + 1))
    FAILED+=("$name — $reason")
    CHECK_NAMES+=("$name")
    CHECK_STATUS+=("fail")
    CHECK_MS+=("$ms")
    CHECK_REASON+=("$reason")
}

hr() { printf "%s%s%s\n" "$D" "$(printf '─%.0s' {1..76})" "$Z"; }

# ---- HTTP helpers --------------------------------------------------------
# get_json URL — echo body if HTTP 200, otherwise fail; also print body to stderr for debug
get_status() {
    local url="$1"
    curl -sS -o /dev/null -w '%{http_code}' -m 30 "$url"
}

get_body() {
    local url="$1"
    curl -sSf -m 30 "$url"
}

post_analyze() {
    local text="$1"
    local body_json
    body_json=$(TEXT="$text" python3 -c 'import json, os; print(json.dumps({"text": os.environ["TEXT"]}))')
    curl -sS -m 30 -X POST "$TARGET_URL/analyze" \
        -H 'Content-Type: application/json' \
        -d "$body_json"
}

json_field() {
    local field="$1"
    python3 -c "
import json, sys
d = json.load(sys.stdin)
for k in '''$field'''.split('.'):
    d = d[k]
print(d)
"
}

# ---- Individual test functions ------------------------------------------

check_ready() {
    local name="/ready returns 200 with ready=true"
    local start; start=$(now_ms)
    local body; body=$(get_body "$TARGET_URL/ready" 2>&1)
    local end; end=$(now_ms)
    if echo "$body" | grep -q '"ready":true'; then
        record_pass "$name" "$((end - start))"
    else
        record_fail "$name" "$((end - start))" "unexpected body: ${body:0:120}"
    fi
}

check_gpu() {
    local name="/gpu returns valid structure"
    local start; start=$(now_ms)
    local body; body=$(get_body "$TARGET_URL/gpu" 2>&1)
    local end; end=$(now_ms)
    local missing=""
    for key in device_setting using_gpu torch_installed cuda_available devices; do
        if ! echo "$body" | grep -q "\"$key\""; then
            missing="$missing $key"
        fi
    done
    if [[ -n "$missing" ]]; then
        record_fail "$name" "$((end - start))" "missing keys:$missing"
    else
        record_pass "$name" "$((end - start))"
    fi
}

check_health_deep() {
    local name="/health returns 200 with llm.ok and local.ok"
    local start; start=$(now_ms)
    local code; code=$(get_status "$TARGET_URL/health")
    local body; body=$(get_body "$TARGET_URL/health" 2>&1)
    local end; end=$(now_ms)
    if [[ "$code" != "200" ]]; then
        record_fail "$name" "$((end - start))" "HTTP $code: ${body:0:200}"
        return
    fi
    if echo "$body" | grep -q '"llm":{"ok":true' && echo "$body" | grep -q '"local":{"ok":true'; then
        record_pass "$name" "$((end - start))"
    else
        record_fail "$name" "$((end - start))" "one dependency reported not ok: ${body:0:200}"
    fi
}

# check_sentiment "check name" "input text" "expected: positive|negative|neutral"
check_sentiment() {
    local name="$1" text="$2" expected="$3"
    local start; start=$(now_ms)
    local body; body=$(post_analyze "$text")
    local end; end=$(now_ms)

    local llm_s local_s
    llm_s=$(echo "$body" | json_field llm.sentiment 2>/dev/null || echo "")
    local_s=$(echo "$body" | json_field local.sentiment 2>/dev/null || echo "")

    if [[ "$llm_s" == "$expected" && "$local_s" == "$expected" ]]; then
        record_pass "$name (both → $expected)" "$((end - start))"
    elif [[ "$llm_s" == "$expected" || "$local_s" == "$expected" ]]; then
        record_fail "$name" "$((end - start))" "only one model got it right (llm=$llm_s, local=$local_s, expected=$expected)"
    else
        record_fail "$name" "$((end - start))" "neither model got it right (llm=$llm_s, local=$local_s, expected=$expected)"
    fi
}

check_validation() {
    local name="/analyze without text returns 422"
    local start; start=$(now_ms)
    local code; code=$(curl -sS -o /dev/null -w '%{http_code}' -m 15 -X POST "$TARGET_URL/analyze" \
        -H 'Content-Type: application/json' -d '{}')
    local end; end=$(now_ms)
    if [[ "$code" == "422" ]]; then
        record_pass "$name" "$((end - start))"
    else
        record_fail "$name" "$((end - start))" "expected 422, got $code"
    fi
}

check_response_shape() {
    local name="/analyze response has all required fields with valid types"
    local start; start=$(now_ms)
    local body; body=$(post_analyze "This is a test message.")
    local end; end=$(now_ms)
    local ok=1 msg=""
    for field in text llm.sentiment llm.confidence llm.reasoning llm.model \
                 local.sentiment local.confidence local.model local.device \
                 agreement; do
        if ! echo "$body" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for k in '''$field'''.split('.'):
    d = d[k]
" >/dev/null 2>&1; then
            ok=0; msg="missing/invalid: $field"
            break
        fi
    done
    if (( ok )); then
        # Confidence must be numeric in [0, 1]
        for cf in llm.confidence local.confidence; do
            local val
            val=$(echo "$body" | json_field "$cf")
            if ! python3 -c "import sys; v = float('$val'); sys.exit(0 if 0 <= v <= 1 else 1)"; then
                ok=0; msg="$cf=$val out of [0, 1]"
                break
            fi
        done
    fi
    if (( ok )); then
        record_pass "$name" "$((end - start))"
    else
        record_fail "$name" "$((end - start))" "$msg"
    fi
}

check_unicode() {
    local text='C'\''est fantastique! 🎉 Amazing.'
    check_sentiment "/analyze handles unicode + emoji" "$text" "positive"
}

check_long_text() {
    local text
    text=$(python3 -c "print(('The service was outstanding. I loved everything. ' * 30).strip())")
    local start; start=$(now_ms)
    local body; body=$(post_analyze "$text")
    local end; end=$(now_ms)
    local llm_s; llm_s=$(echo "$body" | json_field llm.sentiment 2>/dev/null || echo "")
    if [[ "$llm_s" == "positive" ]]; then
        record_pass "/analyze handles long text (~1500 chars)" "$((end - start))"
    else
        record_fail "/analyze handles long text (~1500 chars)" "$((end - start))" "unexpected sentiment=$llm_s"
    fi
}

check_special_chars() {
    local text='He said "amazing job!!!" — with quotes & <html> and \n newlines?'
    local start; start=$(now_ms)
    local body; body=$(post_analyze "$text")
    local end; end=$(now_ms)
    if echo "$body" | grep -q '"sentiment"'; then
        record_pass "/analyze handles special chars (quotes, html, escapes)" "$((end - start))"
    else
        record_fail "/analyze handles special chars (quotes, html, escapes)" "$((end - start))" "response missing sentiment: ${body:0:200}"
    fi
}

# ---- Stress tests --------------------------------------------------------
run_stress() {
    printf "\n%s[stress] concurrency + throughput%s\n" "$B$Y" "$Z"
    local N=10
    printf "  ${D}sending %d concurrent /analyze requests...${Z}\n" "$N"
    local start; start=$(now_ms)
    local ok=0 total=0
    local body_json
    body_json='{"text":"quick concurrent test"}'
    for i in $(seq 1 $N); do
        (
            code=$(curl -sS -o /dev/null -w '%{http_code}' -m 30 -X POST "$TARGET_URL/analyze" \
                -H 'Content-Type: application/json' -d "$body_json")
            echo "$code"
        ) &
    done
    local codes=""
    while read -r c; do codes="$codes $c"; done < <(wait; jobs -p 2>/dev/null)
    # Bash wait+backgrounded jobs pattern is racy; do a simpler collect
    codes=""
    for i in $(seq 1 $N); do
        code=$(curl -sS -o /dev/null -w '%{http_code}' -m 30 -X POST "$TARGET_URL/analyze" \
            -H 'Content-Type: application/json' -d "$body_json" 2>/dev/null &
            wait $!
            )
        [[ "$code" == "200" ]] && ok=$((ok + 1))
        total=$((total + 1))
    done
    local end; end=$(now_ms)
    local ms=$((end - start))
    if [[ "$ok" == "$N" ]]; then
        record_pass "stress: $N sequential /analyze all 200" "$ms"
    else
        record_fail "stress: $N sequential /analyze all 200" "$ms" "$ok/$N succeeded"
    fi
    local per_req=$((ms / N))
    printf "  ${D}avg response time: %dms per request, throughput ~ %.1f req/s${Z}\n" \
        "$per_req" "$(python3 -c "print(1000.0 * $N / $ms if $ms else 0)")"
}

# ---- Run tests -----------------------------------------------------------

printf "\n%sintegration-test.sh%s   target=%s\n" "$B$Y" "$Z" "$TARGET_URL"
hr

printf "\n${B}Basic contract${Z}\n"
check_ready
check_gpu
check_health_deep

printf "\n${B}Sentiment correctness${Z}\n"
check_sentiment "positive: enthusiastic short"          "I loved the movie, it was fantastic!"                        positive
check_sentiment "negative: complaint short"             "The service was slow and the food was cold."                 negative
check_sentiment "neutral: factual statement"            "The package arrived Tuesday afternoon."                      neutral
check_sentiment "positive: subtle multi-clause"         "Everything went smoothly and the outcome exceeded expectations." positive
check_sentiment "negative: subtle multi-clause"         "The service was somewhat lacking, though it had potential."  negative

printf "\n${B}Robustness${Z}\n"
check_validation
check_unicode
check_long_text
check_special_chars
check_response_shape

if (( INCLUDE_STRESS )); then
    run_stress
fi

# ---- Summary -------------------------------------------------------------
TOTAL=$((PASS + FAIL))
printf "\n"; hr
printf "%d checks  ${G}%d passed${Z}  ${R}%d failed${Z}\n\n" "$TOTAL" "$PASS" "$FAIL"

# JSON output — arrays exported as env vars, python zips them into records.
if [[ -n "$JSON_OUT" ]]; then
    NAMES_JOINED=$(IFS=$'\x1f'; echo "${CHECK_NAMES[*]}")
    STATUS_JOINED=$(IFS=$'\x1f'; echo "${CHECK_STATUS[*]}")
    MS_JOINED=$(IFS=$'\x1f'; echo "${CHECK_MS[*]}")
    REASON_JOINED=$(IFS=$'\x1f'; echo "${CHECK_REASON[*]}")
    TARGET_URL="$TARGET_URL" \
    NAMES="$NAMES_JOINED" \
    STATUSES="$STATUS_JOINED" \
    MSS="$MS_JOINED" \
    REASONS="$REASON_JOINED" \
    JSON_OUT="$JSON_OUT" \
    PASSED="$PASS" FAILED_COUNT="$FAIL" TOTAL="$TOTAL" \
    python3 - <<'PY'
import json, os
SEP = "\x1f"
names    = os.environ["NAMES"].split(SEP)    if os.environ["NAMES"]    else []
statuses = os.environ["STATUSES"].split(SEP) if os.environ["STATUSES"] else []
mss      = [int(x) for x in os.environ["MSS"].split(SEP)] if os.environ["MSS"] else []
reasons  = os.environ["REASONS"].split(SEP)  if os.environ["REASONS"]  else []
checks = [
    {"name": n, "status": s, "ms": m, "reason": r or None}
    for n, s, m, r in zip(names, statuses, mss, reasons)
]
with open(os.environ["JSON_OUT"], "w") as f:
    json.dump({
        "target":  os.environ["TARGET_URL"],
        "passed":  int(os.environ["PASSED"]),
        "failed":  int(os.environ["FAILED_COUNT"]),
        "total":   int(os.environ["TOTAL"]),
        "checks":  checks,
    }, f, indent=2)
PY
    printf "wrote %s\n\n" "$JSON_OUT"
fi

if (( FAIL > 0 )); then
    printf "${R}Failed:${Z}\n"
    for f in "${FAILED[@]}"; do
        printf "  - %s\n" "$f"
    done
    printf "\n${Y}Common causes:${Z}\n"
    printf "  - Not on BYU VPN\n"
    printf "  - Staging deploy hasn't finished yet (Coolify's /health is still red)\n"
    printf "  - LLM upstream (LiteLLM on rigel:4000) is down\n"
    printf "  - Recent code change altered response shape\n"
    exit 1
fi

printf "${G}Green across the board — safe to promote to main.${Z}\n"
exit 0
