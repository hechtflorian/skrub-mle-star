# MLE-STAR + skrub — improvement backlog

Latest run: `adk_run_20260611_165054.log` · workspace `california-housing-prices/`

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

## Next up (prioritized)

### P1 — Tuning search → bake integrity (**highest priority**)

**Goal:** Never pretend tuning succeeded when search failed; define an explicit fallback.

- [ ] `check_tune_bake_finish`: require non-empty `tune_best_params_{task_id}` **OR** explicit skip flag when search exhausted
- [ ] `check_bug_existence` for `tune_implement`: treat `returncode==0` without parsed `tune_best_params` as still broken
- [ ] On search failure after all rollback/debug: **skip bake** or bake **`tune_plan.tunable_params[*].default`** literals (not silent structural re-copy)
- [ ] Keep tune **fail-fast** (do **not** revert to 10 blind implement retries)

**Rationale:** Fail-fast saved LLM time; the gap is bake running with no handoff. Fallback should be **plan defaults**, not arbitrary structural params.

### P1b — Refinement implement fail-fast (optional, mirrors tuning)

- [ ] `check_plan_implement_finish`: after **first exec attempt** without success, exit run loop → debug (same pattern as `tune_implement`)
- [ ] Alternative/complement: pass last `stderr` snippet into implement agent on retry (heavier)

**Rationale:** 10 blind implement calls on the same DataOp error wasted ~10 LLM rounds; debug fixed it immediately.

### P1c — Skill: eager ops on DataOp graph objects (#18)

- [ ] `common_failure_fixes.md` #18: no `"col" in X.columns`, `.columns` membership, or pandas `.drop()` on marked `X` DataOp — use `.drop(..., errors="ignore")` on `data_train` before `.mark_as_X()` or skrub selectors
- [ ] One line in `refinement/prompt.py` `IMPLEMENT_PLAN_INSTR` pointing to #18

**Evidence:** `plan_implement` loop error in `adk_run_20260611_165054.log` L1548–1562.

### P1d — Debug backbone drift (refinement + tuning)

- [ ] Strengthen plan_implement debug: keep **same estimator class** as structural code (Ridge swap regressed to 71685)
- [ ] Tune debug already has preserve-search line; still violated — consider deterministic `get_run_code_condition` only (partially done)

### P4 — Lower priority / hygiene

- [ ] `init_plan`: validate `refine_code_block` is a patch, not full script (P4)
- [ ] `analyze_run.py`: auto-link latest `run-logs/adk_run_*.log` when given workspace path only
- [ ] Early-stage Block 2 still absent in this run (good); keep monitoring

---

## Reference scores

| Run | Init | Refinement best | Tuning | Submission |
|-----|------|-----------------|--------|------------|
| 20260610 | 51584.71 | 51584.71 (0 Δ) | n/a / defaults | 51269.10 |
| **20260611** | **54645.59** | **51830.74** | n/a / structural bake | **51830.74** |

---

## P0 implementation plan (archived — verified 2026-06-11)

P0 fixed silent tool-only pass. Confirmed: initial implement now outputs code and changes score. Inner implement still hits vanilla 10× retry before debug — see P1b.
