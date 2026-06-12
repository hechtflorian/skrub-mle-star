# MLE-STAR + skrub — improvement backlog

Latest run: `adk_run_20260611_165054.log` · workspace `california-housing-prices/`
Work one item off, check when done.
---

## Completed (implemented)

| ID | What | Files |
|----|------|-------|
| **P0** | Refinement no-op pass — block tool-only / unchanged re-eval | `debug_util.py`, `refinement/agent.py`, `refinement/prompt.py` |
| **P2** | Tuning `choose_*` inline pattern + search contract enforcement | `choices_hparam_pattern.md`, `common_failure_fixes.md` #16, `tuning/prompt.py`, `code_util.py`, `debug_util.py` |
| **P3** | Ablation `.apply_func` / DataOp helper guidance | `common_failure_fixes.md` #17, `refinement/prompt.py` (ablation) |
| **Tune fail-fast** | `check_tune_implement_finish`: one implement attempt → debug | `tuning/agent.py` |

---

## Latest run analysis (2026-06-11)

**Scores (holdout RMSE):**

| Stage | RMSE | Notes |
|-------|------|-------|
| Init `train0` | 54645.59 | CatBoost baseline |
| Refinement initial implement | **51830.74** | Ratio FE — **−2815 vs init** |
| Refinement inner implement (post-debug) | 71685.11 | Ridge drift — **not promoted** |
| Tuning search | n/a | Never succeeded |
| Tuning bake / `train1` | 51830.74 | **Structural copy**, not searched params |
| Submission | 51830.74 | Same as promoted structural |
| `tune_winner_source` | `structural` | Correct |

### What worked

- **P0 verified:** `plan_implement_initial` emitted code (log L499); ratio features landed; real improvement vs init.
- **P3 / ablation:** Clean holdout ablation with `@skrub.deferred` ratios; useful signal (ratios ≈ −2.8k RMSE).
- **Promotion logic:** Inner-loop Ridge regression (71685) did **not** overwrite the good CatBoost solution (51830).
- **Tune fail-fast:** Debug stepped in quickly (not 20 blind implement retries); `get_run_code_condition` blocked RF/no-search scripts from “passing” exec.

### What still failed

**Refinement — `plan_implement_agent` (10 implement retries, then debug)**

- Log: **10** `[plan_implement_agent_1]: ```python` blocks before `plan_implement_bug_summary_agent`.
- Root error: `"households" in X_train.columns` — **membership on DataOp**, not pandas (`train0_improve1.py`).
- Implement loop uses vanilla **`max_retry=10`** with **no stderr in implement prompt** → blind retries of the same class of bug.
- Debug fixed in one shot but **swapped CatBoost → Ridge** (backbone drift; debug prompt not fully respected).

**Tuning — search never completed; bake used structural defaults**

- First `tune_implement` still used `lr = skrub.choose_float(...)` → **NumericChoice** (#16 not followed despite docs).
- Fail-fast → many debug rounds; debug **removed search**, drifted to **RandomForest**, never produced compliant script.
- `train_code_tune_search_exec_result_1: {}`, **`tune_best_params`: absent**.
- `tune_bake` baked **structural** CatBoost (`iterations=5000, lr=0.03, depth=8`) — same RMSE as `train1`; **not** tune-plan search defaults.
- **P1 still open:** bake runs without successful search handoff.

---

## Implemented 2026-06-12 (P1 + P1b + P1c) — pending rerun validation

### Root cause discovered (verified by repro, skrub 0.9.0)

The r7 tune search failure was **not** prompt non-compliance: skrub resolves inline
`choose_*` in estimator constructor kwargs **only for `sklearn.base.BaseEstimator`
subclasses** (sklearn, LightGBM, XGBoost). CatBoost is not one → `NumericChoice`
leaks into `fit` → JSON crash, even with the previously documented inline pattern
(#16). Working alternative (verified): `choose_from({label: Estimator(**params)})`
variant grid; winner via `search.results_.iloc[0][<choice_name>]`. This also
explains debug backbone drift (model swap was the only "fix" that could run).
See `docs/lessons.md`.

### P1 — Tuning search → bake integrity

- [x] Skill: `choices_hparam_pattern.md` — sklearn-API rule + Pattern 4 (`choose_from` variant grid for non-sklearn estimators, verified snippet)
- [x] Skill: `common_failure_fixes.md` #16 rewritten (mechanism fix, never swap model class)
- [x] `tuning/prompt.py`: plan + implement lines for non-sklearn backbones (model-agnostic, points to Pattern 4)
- [x] `code_util.evaluate_code`: `tune_implement` exiting 0 without parseable non-empty `TUNING_BEST_PARAMS` → synthetic failure (rc=1 + contract stderr) so debug/rollback engage
- [x] `code_util.map_tuning_best_params`: identity guard when raw keys already match plan names (positional remap would scramble values)
- [x] `tuning/agent.py` `prepare_tune_bake_inputs` (before bake): source = `search` | `plan_defaults` (from `tunable_params[*].default`) | `skipped` (sentinel exec result, zero LLM calls); state key `tune_param_source_{task_id}`
- [x] `check_tune_bake_finish`: gate restored (rc==0 + score + code + best_params + no placeholders), early-exit on skip sentinel
- [x] Tune fail-fast kept (no revert to blind implement retries)

### P1b — Eager ops on DataOp graph objects

- [x] `common_failure_fixes.md` #18 added (deferred-helper or `errors="ignore"` fixes)
- [x] One pointer line in `refinement/prompt.py` `IMPLEMENT_PLAN_INSTR`

### P1c — Debug backbone drift

- [x] `debug_prompt.py`: backbone line strengthened — never swap model family; fix `choose_*` mechanism per #16; preserve search contract
- [ ] Refinement implement retries: user adapts `config.py` `max_retry` (decided against code fail-fast)

### Rerun validation checklist (next run)

- [ ] `train_tune_search.py` uses Pattern 4 if backbone is CatBoost; search completes
- [ ] `tune_best_params_{task}` has human-readable names; `tune_param_source` = `search`
- [ ] No backbone drift in tune debug rounds (same model family throughout)
- [ ] On forced search failure: bake uses plan defaults (`tune_param_source: plan_defaults`), no silent structural re-copy
- [ ] Refinement inner implement: #18 error class gone or debug fixes it in ≤1 round

---

## Run analysis 2026-06-12 (`adk_run_20260612_144354`) + second pass (P5/P7/anti-drift)

**Scores:** init 54228.68 → refinement initial implement **52047.99** (ratio+geo FE) → inner implement 58101.88 (HGB drift, not promoted) → tuning skipped (search 2× 600s timeout) → ensemble1 **51971.31** = submission.

**P1 verdict (previous pass): worked.** Tune plan + implement used Pattern 4 (`choose_from` CatBoost grid, plan rationale cites BaseEstimator); zero NumericChoice crashes; no model swap in tune debug; honest skip on search failure (`tune_param_source: "skipped"`, no fake bake, structural promoted).

**New findings:**
1. Tune search now fails on **compute, not correctness**: 4 CatBoost@4500-iter fits cannot finish in `exec_timeout=600` (one structural fit ≈ 195s). Two timeouts ≈ 20 wasted minutes.
2. **Debug drift confirmed as root problem** (log L353–595): ablation agent produced a *good* CatBoost variant script with one trivial bug (`train_df` not loaded); the debug agent rewrote it from scratch — single HGB fit, variants gone, Block 2 added. Drift comes from debug regenerating instead of patching.
3. Plan-defaults fallback never engages for `choose_from` grids (no `default` keys in plan schema) — degraded to skip, which was acceptable.

### Implemented (second pass 2026-06-12)

- [x] **P5** search compute budget — `choices_hparam_pattern.md` budget rule (reduced capacity during search, `n_jobs=1` for internally-threaded estimators, (n_iter+1)×fit-time < timeout); `tuning/prompt.py` plan + implement budget lines; bake restores structural capacity
- [x] **P7 (minimal)** ablation contract — both ablation prompts pin `Ablation[<variant>] <metric>: <value>` per-variant print; `evaluate_code` fails ablation runs with <2 such lines (synthetic rc=1 + contract stderr)
- [x] **Anti-drift** — `BUG_REFINE_INSTR`: smallest-possible-change rule (no regeneration, no removing variants/features/search, no adding test/submission stages); `debug_util` injects deterministic backbone contract (estimator classes regex-extracted from buggy code) into every debug instruction
- [ ] **P6 deferred** (deterministic Block 2 gates) — needs more evidence across runs before adding exec gates

### Rerun validation checklist (next run)

- [ ] Tune search completes inside timeout (reduced iterations during search); `tune_param_source: search`
- [ ] Baked code restores structural capacity for non-tuned params
- [ ] Ablation keeps backbone + ≥2 `Ablation[...]` lines even after debug rounds
- [ ] Debug fixes are minimal diffs (no model swaps, no Block 2 additions)
- [ ] Consider adding a second (classification, high-cardinality) task for generalization testing

---

## Implemented 2026-06-12 (third pass) — ContextWindowExceededError fix

**Root cause (deterministic):** `init_plan` returned prose → `refine_code_block` stored as
`""` → `plan_implement` merge ran `prev_code.replace("", new_code)`, which inserts the
new code between *every character* of the old script (8 MB string) → injected into the
debug prompt → 1.69M tokens. Verified by repro; see `docs/lessons.md`.

- [x] `debug_util.get_code_from_response`: empty `code_block` → no-op (never `replace("")`); existing tool-call-only guards untouched (verified by test)
- [x] `refinement/agent.check_plan_implement_finish`: empty `refine_code_block` → skip implement stage entirely (immediate `LlmResponse()`, zero LLM calls; outer loop keeps previous solution via existing `"score" not in exec_result` fallback)
- [x] `code_util.truncate_for_state`: stdout/stderr/ablation_result bounded (2k head + 4k tail) **after** all extraction (score, `TUNING_BEST_PARAMS`, ablation gate run on full text); tail keeps metric lines + tracebacks
- [x] `SKILL.md` hard constraint: silence training logs (CatBoost `verbose=0`, LGBM `verbose=-1`, XGB `verbosity=0`) — prevents the bloat at the source
- [x] `debug_prompt.py`: load at most 1-2 skill references (was "at least 2")
- [x] `debug_util` backbone contract tightened: wrong import → fix the import for the same class, never substitute a model family
- [x] Both extract-plan prompts: `code_block` must be verbatim contiguous excerpt (not full script, no fences); response JSON-only, never prose
- [x] `common_failure_fixes.md` #20: no custom wrapper classes around DataOps graphs; combine predictions, not learner objects (verified pattern from this run's `train0.py`)

### Rerun validation (context fix)

- [ ] No `ContextWindowExceededError`; `final_state.json` stays small (no multi-MB stdout)
- [ ] `init_plan` returns valid JSON block (or implement stage skips cleanly with zero LLM calls)
- [ ] plan_implement agents still output final code after skill tool calls (regression check on the tool-call-only fix)

---

## Implemented 2026-06-12 (fourth pass) — run 165943 analysis + FE-drop fix + evaluation tooling

**Run 165943 verdict:** all third-pass fixes held (no context explosion, `final_state.json` 86 KB, ablation contract met, tune search completed with Pattern 4, debug kept backbone across 3 tuning rounds, honest promotion). One real defect found: **tuning dropped the structural FE** — search/bake graphs were bare `TableVectorizer+CatBoost`, baked score = exactly init (54886.72), so the stage could never beat structural (52062.02). Root cause: refinement's promoted `train1.py` is a merged runtime variant-chooser script; tune implement simplified instead of preserving it.

- [x] `refinement/prompt.py` `IMPLEMENT_PLAN_INSTR`: implement plan as one resolved pipeline; no runtime variant-chooser loops (variant comparison belongs to ablation)
- [x] `tuning/prompt.py` `TUNE_IMPLEMENT_INSTR`: reproduce structural pipeline verbatim (deferred FE funcs, apply_func steps, scalers, encoders); if structural compares variants at runtime, reproduce only the winner
- [x] `tuning/prompt.py` `TUNE_BAKE_INSTR`: carry over full pipeline verbatim, only tuned literals differ
- [x] `analyze_run.py` improved: fixed mislabeled rows (improve iters now enumerated dynamically; "promoted (train1)" labeled correctly), shows `tune_param_source`, stage-gain deltas, contract-violation/context-error counts from log, `to_row()` flat export, sentinel scores cleaned, `--json` emits comparison row
- [x] `aggregate_runs.py` (new): walks `submissions/` for `final_state.json` bundles, one row per run (task/variant/run from path), prints table + per-(task,variant) mean±std, optional `--out` CSV; verified on all 23 historical runs

### Rerun validation (next run)

- [ ] `train_tune_search.py` / `train_tune_baked.py` contain the structural FE (deferred funcs + scaler); baked score ≠ init score when refinement improved
- [ ] `train1.py` is a single resolved pipeline (no variant loop, no duplicated imports)

---

## Next up (prioritized)

### P4 — Lower priority / hygiene

- [ ] `init_plan`: validate `refine_code_block` is a patch, not full script (P4)
- [ ] `analyze_run.py`: auto-link latest `run-logs/adk_run_*.log` when given workspace path only
- [ ] Early-stage Block 2 still absent in this run (good); keep monitoring

### Carried-over open items (from working-progress doc)

- [x] Truncate tune-search stdout before state write (`final_state.json` bloat) — done via `truncate_for_state` (third pass)
- [ ] Enable `use_data_leakage_checker=True` once holdout docs stabilize
- [ ] Unified validation harness / same-pipeline ablation for planner trust
- [ ] Feed TableReport profile to planners beyond ablation
- [ ] Evaluation phase: baseline vs skrub-variant runs on ≥10 Kaggle tabular tasks (3 repeats), DataOps adherence + score delta + debug efficiency metrics

---

## Reference scores

| Run | Init | Refinement best | Tuning | Submission |
|-----|------|-----------------|--------|------------|
| 20260610 | 51584.71 | 51584.71 (0 Δ) | n/a / defaults | 51269.10 |
| 20260611 | 54645.59 | 51830.74 | n/a / structural bake | 51830.74 |
| **20260612** | **54228.68** | **52047.99** | skipped (search timeout) | **51971.31** (ensemble) |
| **20260612b (165943)** | 54886.72 | 52062.02 | search ok, but FE dropped → bake lost | **52034.48** (ensemble) |

---

## P0 implementation plan (archived — verified 2026-06-11)

P0 fixed silent tool-only pass. Confirmed: initial implement now outputs code and changes score. Inner implement still hits vanilla 10× retry before debug — see P1b.
