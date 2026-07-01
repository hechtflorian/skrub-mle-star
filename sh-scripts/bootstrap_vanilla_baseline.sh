#!/usr/bin/env bash
# Bootstrap a vanilla MLE-STAR baseline from git history (default: commit ffa365c).
#
# ffa365c includes: ChatAI model routing, DDG web search, GPT-5 temperature fix,
# common_util text=None guard. No ADK skills, TableReport, or tuning stage.
# Prompts at ffa365c lightly encourage skrub DataOps in prose (no skill toolset).
#
# Usage (from mle-star_improved repo root):
#   ./scripts/bootstrap_vanilla_baseline.sh
#   ./scripts/bootstrap_vanilla_baseline.sh --dest ../mle-star_vanilla --revert-prompts
#
# Recommended: git worktree mode (default) — branch vanilla-baseline @ ffa365c in a
# sibling directory, same repo history, easy diff vs improve-refinement.

set -euo pipefail

BASELINE_COMMIT="ffa365c"
PROMPT_COMMIT="6c96e03"
BRANCH_NAME="vanilla-baseline"
MODE="worktree"          # worktree | export
DEST=""
REVERT_PROMPTS=0
REMOVE_SKRUB_DEP=0
DRY_RUN=0
SYNC_TASKS=0

AGENTS_REL="agents/machine-learning-engineering"
MLE_REL="${AGENTS_REL}/machine_learning_engineering"

usage() {
  cat <<'EOF'
Bootstrap vanilla MLE-STAR baseline from git history.

Options:
  --dest PATH           Output directory (default: ../mle-star_vanilla)
  --baseline-commit SHA Baseline snapshot (default: ffa365c)
  --mode MODE           worktree (default) or export
  --revert-prompts      Restore prompt.py files from 6c96e03 (pure sklearn wording)
  --remove-skrub-dep    Drop skrub/optuna lines from pyproject.toml (use with --revert-prompts)
  --sync-tasks          Copy tasks/ from current HEAD into the baseline tree
  --dry-run             Print actions only
  -h, --help            Show this help

Modes:
  worktree  Create branch vanilla-baseline at baseline commit + git worktree (recommended)
  export    Copy agents/machine-learning-engineering tree only (no git link)

After bootstrap:
  cd <dest>/agents/machine-learning-engineering
  cp ../../mle-star_improved/agents/machine-learning-engineering/.env .   # or symlink
  uv sync && uv run adk run machine_learning_engineering
EOF
}

log() { printf '==> %s\n' "$*"; }
run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    eval "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --baseline-commit) BASELINE_COMMIT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --revert-prompts) REVERT_PROMPTS=1; shift ;;
    --remove-skrub-dep) REMOVE_SKRUB_DEP=1; shift ;;
    --sync-tasks) SYNC_TASKS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "Error: run from inside the mle-star_improved git repository." >&2
  exit 1
fi
cd "$REPO_ROOT"

if [[ -z "$DEST" ]]; then
  DEST="$(dirname "$REPO_ROOT")/mle-star_vanilla"
fi
DEST="$(realpath -m "$DEST")"

if ! git cat-file -e "${BASELINE_COMMIT}^{commit}" 2>/dev/null; then
  echo "Error: baseline commit not found: ${BASELINE_COMMIT}" >&2
  exit 1
fi

BASELINE_SHA="$(git rev-parse "${BASELINE_COMMIT}")"
log "Repo:     ${REPO_ROOT}"
log "Baseline: ${BASELINE_COMMIT} (${BASELINE_SHA})"
log "Mode:     ${MODE}"
log "Dest:     ${DEST}"

case "$MODE" in
  worktree)
    if [[ -e "$DEST" ]]; then
      echo "Error: dest already exists: ${DEST}" >&2
      echo "Remove it or pick another --dest path." >&2
      exit 1
    fi
    if git show-ref --verify --quiet "refs/heads/${BRANCH_NAME}"; then
      log "Branch ${BRANCH_NAME} already exists — adding worktree at ${BASELINE_SHA}"
      run "git worktree add \"${DEST}\" \"${BRANCH_NAME}\""
    else
      run "git worktree add -b \"${BRANCH_NAME}\" \"${DEST}\" \"${BASELINE_SHA}\""
    fi
    TARGET_ROOT="$DEST"
    ;;
  export)
    run "mkdir -p \"${DEST}\""
    run "git archive \"${BASELINE_SHA}\" \"${AGENTS_REL}\" | tar -x -C \"${DEST}\""
    TARGET_ROOT="$DEST"
    ;;
  *)
    echo "Error: unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

AGENTS_DIR="${TARGET_ROOT}/${AGENTS_REL}"
MLE_DIR="${TARGET_ROOT}/${MLE_REL}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run complete (no files written)."
  cat <<EOF

Would run post-bootstrap steps:
  - clean workspace/ and run-logs/
  - revert prompts: ${REVERT_PROMPTS}
  - remove skrub deps: ${REMOVE_SKRUB_DEP}
  - sync tasks from HEAD: ${SYNC_TASKS}
  - write VANILLA_BASELINE.md and verify tree

Re-run without --dry-run to apply.
EOF
  exit 0
fi

if [[ ! -d "$AGENTS_DIR" ]]; then
  echo "Error: expected agents dir missing: ${AGENTS_DIR}" >&2
  exit 1
fi

cleanup_workspace() {
  local ws="${MLE_DIR}/workspace"
  if [[ -d "$ws" ]]; then
    log "Cleaning stale workspace under ${ws}"
    run "find \"${ws}\" -mindepth 1 -maxdepth 1 -exec rm -rf {} +"
  fi
  local logs="${AGENTS_DIR}/run-logs"
  if [[ -d "$logs" ]]; then
    run "find \"${logs}\" -name 'adk_run_*.log' -delete 2>/dev/null || true"
  fi
}

revert_prompt_file() {
  local rel="$1"
  local full="${MLE_DIR}/${rel}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] revert prompt %s from %s\n' "$rel" "$PROMPT_COMMIT"
    return
  fi
  git show "${PROMPT_COMMIT}:${AGENTS_REL}/machine_learning_engineering/${rel}" > "$full"
}

revert_prompts() {
  log "Reverting prompts to ${PROMPT_COMMIT}"
  revert_prompt_file "sub_agents/initialization/prompt.py"
  revert_prompt_file "sub_agents/refinement/prompt.py"
  revert_prompt_file "sub_agents/ensemble/prompt.py"
  revert_prompt_file "sub_agents/submission/prompt.py"
  revert_prompt_file "shared_libraries/debug_prompt.py"
  revert_prompt_file "shared_libraries/data_leakage_prompt.py"
}

remove_skrub_deps() {
  local pyproject="${AGENTS_DIR}/pyproject.toml"
  log "Removing skrub/optuna from ${pyproject}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] strip skrub/optuna from pyproject.toml\n'
    return
  fi
  python3 - <<PY
from pathlib import Path
path = Path("${pyproject}")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
drop = ('"skrub>=', '"optuna>=')
out = [ln for ln in lines if not any(d in ln for d in drop)]
path.write_text("".join(out), encoding="utf-8")
PY
}

sync_tasks_from_head() {
  local src="${REPO_ROOT}/${MLE_REL}/tasks"
  local dst="${MLE_DIR}/tasks"
  if [[ ! -d "$src" ]]; then
    echo "Warning: tasks source missing: ${src}" >&2
    return
  fi
  log "Syncing tasks/ from current HEAD -> baseline tree"
  run "rsync -a --delete \"${src}/\" \"${dst}/\""
}

write_marker() {
  local marker="${TARGET_ROOT}/VANILLA_BASELINE.md"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] write %s\n' "$marker"
    return
  fi
  cat > "$marker" <<EOF
# Vanilla MLE-STAR baseline

- **Baseline commit:** \`${BASELINE_SHA}\` (\`${BASELINE_COMMIT}\`)
- **Branch:** \`${BRANCH_NAME}\` (worktree mode)
- **Prompts reverted to ${PROMPT_COMMIT}:** ${REVERT_PROMPTS}
- **skrub/optuna removed from pyproject:** ${REMOVE_SKRUB_DEP}
- **Created:** $(date -Iseconds)

## What this includes

- OpenAI/ChatAI-compatible model routing (\`ROOT_AGENT_MODEL\`, LiteLLM)
- DuckDuckGo web search for non-Gemini models (\`search_tool_util.py\`)
- GPT-5 temperature compatibility (\`get_compatible_temperature\`)
- Safe response parsing (\`common_util.get_text_from_response\`)

## What this excludes (vs skrub-full on improve-refinement)

- ADK \`skrub-dataops-pipeline\` skill / SkillToolset
- TableReport profiling
- Terminal tuning stage (\`sub_agents/tuning/\`)

## Run

\`\`\`bash
cd agents/machine-learning-engineering
# copy .env from mle-star_improved if needed
uv sync
uv run adk run machine_learning_engineering
\`\`\`

Record this commit SHA in \`experiments/phase1/README.md\` for vanilla runs.
EOF
}

verify_tree() {
  log "Verification"
  local fail=0
  local mle="$MLE_DIR"

  check_absent() {
    if rg -q "$1" "$mle" 2>/dev/null; then
      echo "  FAIL: found forbidden pattern: $1" >&2
      fail=1
    else
      echo "  OK: no $1"
    fi
  }

  check_present() {
    if rg -q "$1" "$mle" 2>/dev/null; then
      echo "  OK: found $1"
    else
      echo "  FAIL: missing expected pattern: $1" >&2
      fail=1
    fi
  }

  check_absent "skill_tool_util"
  check_absent "table_report_util"
  check_absent "sub_agents/tuning"
  check_absent "skills/skrub-dataops-pipeline"
  check_present "search_tool_util"
  check_present "get_compatible_temperature"

  if [[ ! -f "${mle}/shared_libraries/search_tool_util.py" ]]; then
    echo "  FAIL: missing search_tool_util.py" >&2
    fail=1
  fi

  if [[ "$fail" -ne 0 ]]; then
    echo "Verification failed." >&2
    exit 1
  fi
  log "Verification passed"
}

cleanup_workspace

if [[ "$REVERT_PROMPTS" -eq 1 ]]; then
  revert_prompts
  REMOVE_SKRUB_DEP=1
fi

if [[ "$REMOVE_SKRUB_DEP" -eq 1 ]]; then
  remove_skrub_deps
fi

if [[ "$SYNC_TASKS" -eq 1 ]]; then
  sync_tasks_from_head
fi

write_marker
verify_tree

cat <<EOF

Done.

Next steps:
  1. cd ${AGENTS_DIR}
  2. Copy .env from mle-star_improved (same ROOT_AGENT_MODEL / OPENAI_API_*)
  3. uv sync
  4. Smoke: uv run adk run machine_learning_engineering
  5. Archive runs into mle-star_improved/experiments/phase1/<task>/vanilla/...

To compare code vs skrub-full:
  git diff vanilla-baseline..improve-refinement -- ${AGENTS_REL}/machine_learning_engineering

EOF

if [[ "$MODE" == "worktree" ]]; then
  cat <<EOF
To remove worktree later:
  git worktree remove "${DEST}"
  git branch -D ${BRANCH_NAME}   # only if you no longer need the branch

EOF
fi
