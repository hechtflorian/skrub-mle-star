# Refinement + skrub: Operator Handoff (Condensed)

This document summarizes the key design decisions, current behavior, and practical operating guidance for the skrub-integrated refinement stage in MLE-STAR.

## Why this exists

We integrated skrub DataOps skills into refinement to improve solution quality while preserving MLE-STAR’s loop architecture. The goal is to make refinement more structured, more reliable, and less likely to drift into non-DataOps code.

## What changed from vanilla MLE-STAR refinement

### Vanilla (before)
- Refinement primarily acted as procedural code edits around candidate scripts.
- Weaker constraints on pipeline structure allowed drift to sklearn-only orchestration.
- Tool-call-only or partial outputs could slip through in some paths.

### Current skrub-integrated behavior
- DataOps-first expectations:
  - Keep `skrub.var` / `skrub.X` / `skrub.y` (or mark-as-X/y) and `.skb.apply(...)` as main path.
- Skill-backed refinement:
  - `skrub-dataops-pipeline` skill loaded for relevant refinement steps.
  - References loaded conditionally for tuning / encoding / debugging issues.
- Prompt-level constraints:
  - Tool calls are preparation only; code-producing steps must still return runnable Python code in the same response.
- Anti-fake-tuning policy:
  - `choose_*` without an actual search is not “tuned”.

## Current refinement strategy (with skrub)

- **Ablation-first, then refinement**:
  1. Run ablation to identify impactful areas.
  2. Extract target code block and propose plan.
  3. Implement/refine selected block.
- **Bounded search only when justified**:
  - Prefer structural or fixed-parameter changes unless tuning is clearly needed.
  - Keep tuning focused and compact if used.
- **DataOps architecture preservation**:
  - Fix uncertain API usage without rewriting main workflow to sklearn-only.

## How refinement loops work

Refinement is nested loops with fixed iteration budgets from config:

- **Outer loop** (`outer_loop_round`):
  - Performs one full cycle: ablation -> summarize -> init plan -> initial implement -> inner refinement.
- **Inner loop** (`inner_loop_round`):
  - Runs plan refine + plan implement iterations.
- **Debug/rollback loops**:
  - Handle execution failures per stage (`max_debug_round`, `max_rollback_round`, `max_retry` depending on sub-loop).

### Finish conditions (conceptual)
- Ablation stage finishes when ablation script executes successfully.
- Init plan stage finishes when extracted `code_block` is valid and present in source.
- Implement stages currently treat successful execution + parseable score as done.
- Outer-step promotion selects best improvement; if no positive gain, previous solution is carried forward.

## Known pain points still observed

1. **Implement output quality drift**
   - Agents may return incomplete code blocks that run but do not produce meaningful final metric output.
2. **Ablation underuses DataOps levers**
   - Ablation often compares model variants only, while encoding/routing stays default (`TableVectorizer()`).
3. **Sentinel-score risk**
   - If final metric line is missing/unparseable, score can degrade to sentinel behavior in downstream logic.

## What to optimize next (without breaking MLE-STAR logic)

Keep architecture unchanged; improve behavior inside existing stage boundaries.

1. **Ablation dimensions**
   - Add DataOps-aware ablations:
     - encoding strategy variants,
     - selector-based routing (`ApplyToCols` + selectors),
     - compact merge variants.
2. **Plan gating**
   - If ablation indicates structural gains, prioritize structural edits first.
   - Only then use focused tuning where justified.
3. **Output contract strictness (prompt + optional runtime later)**
   - Keep strict same-turn runnable code requirement.
   - Add runtime guard later only if prompt hardening proves insufficient.

## Operator rerun checklist

After each rerun, inspect:

- `final_state.json`
  - `train_code_improve_*` content completeness
  - `train_code_improve_exec_result_*` score validity
  - ablation summaries and extracted code blocks
- workspace scripts:
  - `ablation_0.py`, `train0_improve*.py`, `train1.py`
- run log:
  - tool-call behavior and whether implement stages produce final code + metric lines

Success signals:

- No tool-only or partial implement outputs.
- Ablation tests at least one encoding/preprocessing/routing hypothesis when appropriate.
- Improvement candidates include valid final metric lines and non-sentinel scoring.