# MLE-STAR + skrub: Future Improvement Plan

Handoff document for human operators and coding agents. Use this when proposing or implementing the next iteration of the MLE-STAR pipeline under `agents/machine-learning-engineering/`.

**Related docs:** `WORKING_PROGRESS_MLE_STAR_SKRUB.md`, `Refinement_skrub_handoff.md`, `MLE_STAR_INIT_REFINEMENT_SKRUB_MAP.md`

---

## Purpose

Capture **honest post-mortem findings** from California Housing runs (especially `adk_run_20260606_223108`) and turn them into a prioritized backlog. Items here are intentionally scoped to fit the existing ADK sequential-agent architecture (`include_contents="none"`, state-based handoff).

---

## Current pipeline (baseline for improvements)

```
init → refinement (structural) → [tuning if tuning_enabled] → ensemble → submission
```

**State handoff:** agents do not share chat memory. They read/write explicit keys in `callback_context.state` (`train_code_*`, `*_exec_result_*`, `ablation_summary_*`, `tune_best_params_*`, etc.). See `final_state.json` after each run.

**Config flags:** `tuning_enabled`, `tuning_n_iter`, `tuning_n_jobs` in `shared_libraries/config.py`.

---

## Findings from recent runs (what actually happened)

### Scores are comparable only within the same script/protocol

- All stages parse the **last** `Final Validation Performance:` line from stdout (`code_util.extract_performance_from_text`).
- Different stages often run **different models, splits, or pipelines** → large score spreads are expected, not necessarily parser bugs.
- **Ablation** does not store a numeric `score` in state (only `ablation_result` text); ablation stdout is often from a different pipeline than `train{N}.py`.
- Latest successful run (`223108`): structural holdout RMSE ~**2664** → ensemble/submission ~**2414**; tuning search ~**57712**, bake ~**11021** → correctly **not promoted** (`tune_winner_source_1: structural`).

### TableReport profile (ablation context)

- Built in `init_outer_loop_states` via `table_report_util.load_table_report_dict` → `ablation_table_report_profile_{task_id}`.
- Injected into ablation prompts only (`refinement/prompt.py` → `{data_profile}`).
- On California Housing: **modest theoretical value**; submission history shows no clear lift from profile alone (e.g. r2 without profile ~50.7k vs r7–r10 with profile ~54–57k). Latest big gain came from **plan implement** (geo features + encoders + LGBM), not from ablation output directly.

### Tuning stage

- **Architecture is sound:** separate module, promotion gate, no change to ensemble handoff keys when tuning loses.
- **Empirical result so far:** neutral-to-negative on California Housing (bad search outcome + 6 min runtime + stdout bloat).
- **Operational issue:** `train_code_tune_search_exec_result_*` stdout can reach **~6 MB** (repeated LightGBM warnings) and inflates `final_state.json`.

### Agent “memory”

- No cross-agent conversational memory (`include_contents="none"` everywhere).
- Partial structured memory exists (`prev_ablations_*`, `refine_plans_*`, latest `bug_summary_*` only).
- Repeating failed approaches is expected unless failures are appended to state and injected into prompts.

---

## Priority backlog

### P0 — Correctness and comparability (do first)

| ID | Improvement | Rationale | Touch points |
|----|-------------|-----------|--------------|
| P0-1 | **Same-pipeline ablation** | Ablation must fork the current `train_code_{step}_{task}` (same model family, same holdout protocol). Today ablation often invents unrelated pipelines (e.g. HistGradientBoosting + `train_test_split`), making summaries misleading for planners. | `refinement/prompt.py`, `refinement/agent.py`, optional runtime check in `code_util.get_run_code_condition` |
| P0-2 | **Unified validation harness** | One shared holdout function (fixed seed, split indices written to state or workspace) used by init, refinement, tuning, ensemble scoring. Stops 58k vs 2.6k apples/oranges and reduces in-sample eval bugs (seen in older submission scripts). | New util e.g. `validation_util.py`; skill template; inject into prompts |
| P0-3 | **Parse/store ablation score** | Write `score` on ablation exec results the same way as train code, or copy parsed score into `ablation_summary_*` metadata. Makes `final_state.json` readable and enables gating. | `code_util.evaluate_code` |
| P0-4 | **Stdout truncation before state storage** | Cap stdout/stderr stored in exec results (e.g. keep head + tail + metric lines). Prevents 6 MB `final_state.json` from tune search. | `code_util.run_python_code` or post-process in `evaluate_code` |

### P1 — Make existing features actually help

| ID | Improvement | Rationale | Touch points |
|----|-------------|-----------|--------------|
| P1-1 | **TableReport → planners, not only ablation** | **Done (2026-06-06).** Profile cached once in `init_outer_loop_states`; shared via `get_profile_from_state`; injected into `init_plan` + `plan_refine` only; column cap (`max_columns=30`) for wide tables. | `table_report_util.py`, `refinement/agent.py`, `refinement/prompt.py` |
| P1-2 | **Experiment ledger in state** | Append-only `experiment_log_{task_id}`: `{stage, action, score, error_snippet}`. Inject last N entries into tune/debug/plan prompts. Addresses “agent repeats failed fix” without enabling full chat memory. | New `handoff_util.py`; `debug_util`, `tuning/agent.py`, `refinement/agent.py` |
| P1-3 | **Ablation → plan contract** | Require ablation summary to cite **which code blocks** were toggled and **delta vs baseline** on the same metric line. Reject summaries that only report “code runs now.” | `refinement/prompt.py` (`SUMMARIZE_ABLATION_INSTR`, `EXTRACT_BLOCK_AND_PLAN_INSTR`) |
| P1-4 | **Tuning hardening** | `verbose=-1` on LightGBM in skill pattern; optional **subsampled search**; smaller `n_iter` default; deterministic `tune_bake` from `tune_best_params_*` (reduce LLM rewrite variance). | `choices_hparam_pattern.md`, `tuning/prompt.py`, optional deterministic bake in `code_util` |

### P2 — Quality and evaluation at scale

| ID | Improvement | Rationale | Touch points |
|----|-------------|-----------|--------------|
| P2-1 | **Cross-stage summary key** | After each major stage, write `stage_summary_{task_id}` (best score, winner path, what failed). Ensemble/submission read it. | Stage `after_agent_callback`s |
| P2-2 | **Versioned tune artifacts** | Don't overwrite `train_tune_search.py` blindly on retry; or write `train_tune_search_{attempt}.py` for debugging. | `code_util`, `tuning/agent.py` |
| P2-3 | **Benchmark matrix** | Run baseline vs current on ≥10 tabular tasks, 3 seeds; track win-rate, adherence, debug rounds, cost — per `WORKING_PROGRESS` experiment plan. | `submissions/`, run manifests |
| P2-4 | **Lower tune retry budget** | Separate `max_tune_implement_round` from general debug budget to avoid 10× implement loops on non-gating failures. | `config.py`, `tuning/agent.py` |

### P3 — Nice to have

- Persist `run_notes.json` in workspace (human-readable audit alongside `final_state.json`).
- Block known-bad patterns in `code_util` before exec (extend tune gates pattern).
- Feed **failure_log** into skill resource selection (“load common_failure_fixes if JSON serialization error seen”).

---

## Anti-patterns to avoid

1. **Full cross-agent chat memory** (`include_contents="default"`) — fights ADK design, blows context, reintroduces tool-vs-code confusion.
2. **Tuning inside refinement** — already split out; keep structural vs terminal search separate.
3. **Trusting ablation score from a different pipeline** — misleads tune_plan (“model capacity” tuning when baseline issue is encoding).
4. **Storing unbounded subprocess stdout in state** — operational debt, slow JSON, hard to diff runs.

---

## Suggested implementation order (for coding agents)

1. P0-4 stdout truncation (quick, unblocks artifact size).
2. P0-1 + P0-3 same-pipeline ablation + ablation score (refinement quality).
3. ~~P1-1 TableReport on planners (low risk).~~ **Done** — see P1-1 row above.
4. P0-2 unified validation harness (larger but high leverage).
5. P1-2 experiment ledger (robustness).
6. P1-4 tuning hardening + re-benchmark on California Housing + one messy categorical task.

**Definition of done for a iteration:** one full `adk run` on California Housing where (a) ablation uses same base pipeline as `train0`, (b) `final_state.json` < 500 KB, (c) tuning either improves holdout or exits fast with ledger entry, (d) submission holdout score uses same split as train1.

---

## Open questions

- Should `train_code_tune_search_exec_result_*` score affect promotion, or only bake score? (Currently implement score is stored but promotion uses bake.)
- Is subsampled tuning acceptable for agent feedback if full-data bake runs once at the end?
- Should TableReport associations drive **mandatory** ratio-feature ablations on high-correlation pairs, or stay advisory?

---

## References (code)

| Area | Path |
|------|------|
| Score parsing | `shared_libraries/code_util.py` — `extract_performance_from_text`, `evaluate_code` |
| TableReport | `shared_libraries/table_report_util.py`, `refinement/agent.py` — `init_outer_loop_states` |
| Refinement loops | `sub_agents/refinement/agent.py` — `update_outer_loop_states`, finish checks |
| Tuning | `sub_agents/tuning/agent.py`, `sub_agents/tuning/prompt.py` |
| Pipeline wiring | `machine_learning_engineering/agent.py` |
| Skill tuning pattern | `skills/skrub-dataops-pipeline/references/choices_hparam_pattern.md` |

---

---

## Implemented (2026-06-06): P1-1 TableReport for planners + FE patterns

- **`get_profile_from_state`** reads cached `ablation_table_report_profile_{task_id}` (no TableReport recompute).
- **`format_ablation_profile(..., max_columns=30)`** truncates column listing on wide tables.
- **`init_plan_agent`** and **`plan_refine_agent`** receive `{data_profile}` with selective-use instruction; `plan_implement` unchanged.

---

*Last updated: 2026-06-06 — P1-1 TableReport for planners implemented.*
