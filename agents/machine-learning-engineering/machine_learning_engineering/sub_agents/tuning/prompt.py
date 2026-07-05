"""Prompts for terminal tuning agents."""

TUNE_PLAN_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- Structural refinement is complete. Propose **one** focused tuning plan using skrub `choose_*` nodes.
- Tuning runs once after all refinement outer loops.

# Current structural solution
```python
{code}
```

# Ablation study summary
{ablation_results}

# Prior structural plan outcomes
{plan_summary}

# Your task
- Decide whether terminal holdout search is worth running, or skip it.
- If skip: set `"skip_tuning": true` with a one-sentence `"skip_reason"` (required). Omit `focus_block` / `tunable_params`.
- Skip when ablation and refine context show **no clear tunable lever** (flat model+encoder ablation, refinement already captured the main gain, ensemble too complex for one-block polish, or only marginal hparam headroom on the current backbone). When uncertain, prefer skip — tuning is polish, not a second refinement loop.
- If not skipping: pick **one** focus block only: `model` or `encoder`/`preprocessing`.
- Prefer `model` when ablation showed capacity or model-side effects; prefer `encoder`/`preprocessing` when its ablation clearly mattered.
- If model focus: at most **2** `choose_*` nodes with tight ranges around current literals.
- If encoder focus: at most **2** `choose_*` nodes; if tuning `TableVectorizer`, plan `choose_from` of whole vectorizers or multiple encoder instances.
- If the backbone estimator is **not** a sklearn-API estimator (`sklearn.base.BaseEstimator` subclass — e.g. CatBoost is not), numeric `choose_*` in its constructor will not resolve; plan a small discrete variant grid via `choose_from` instead (see `choices_hparam_pattern.md` Pattern 4), still with `default` values per param.

# Requirements
- Call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and load `references/choices_hparam_pattern.md` via `load_skill_resource`.
- If `focus_block` is `encoder` or `preprocessing`, also load `references/encoding_skrub.md`.
- List which pipeline parts stay frozen in `frozen`.
- Do not propose new feature engineering, backbone swap, or multiple focus blocks.
- Use the same holdout split as the current solution (`train_test_split` size and `random_state`).
- Use the same backbone estimator class(es) as the input Python solution above for your tuning plan; do not substitute a different model family.
- If the structural solution is an ensemble (multiple estimators blended), pick **one** leg to tune (highest expected impact from ablation) and list the other leg(s) in `frozen` with fixed structural params.

# Search budget
- `n_iter={n_iter}`. Holdout only (same split as structural). No CV.
- `make_randomized_search(..., n_jobs=...)`: use **`n_jobs=1`** when the backbone is internally multithreaded (e.g. LightGBM, CatBoost, XGBoost) or the estimator sets `n_jobs` ≠ 1. Otherwise **`n_jobs=2`** is a reasonable default for fast single-threaded sklearn fits; use **`n_jobs=4`** at most when each trial is very cheap.
- Reduce boosted-tree `iterations`/`n_estimators` to ~1/2 of the structural value during search.
- Keep total search time under the execution timeout ({exec_time} seconds): runtime ≈ `(n_iter + 1) * per-trial fit time`.

# Response format
- Return a single JSON object only (no markdown fences, no extra text).

Use this JSON schema:
TunePlan = {{
  "skip_tuning": bool,
  "skip_reason": str,
  "focus_block": "model" | "encoder" | null,
  "rationale": str,
  "tunable_params": [{{"name": str, "kind": "choose_from" | "choose_int" | "choose_float", ...}}],
  "frozen": [str],
  "n_iter": int
}}
Return: TunePlan"""

TUNE_IMPLEMENT_INSTR = """# Introduction
- Implement the terminal tuning plan on the structural solution below.
- This should be a simple task; you should just insert the `choose_*` nodes, run the search, and handover the best params to the next agent.
- Run **real** holdout randomized search with in-graph `choose_*` nodes, then report the best holdout validation score.
- Follow `references/choices_hparam_pattern.md` for the tuning search contract.

# Structural solution
```python
{code}
```

# Tuning plan (JSON)
{tune_plan}

# Requirements
- Load `references/tuning_dataops_template.md` and `references/choices_hparam_pattern.md` via skill tools before editing. Follow the tuning template skeleton; copy structural FE, encoders, and ensemble scoring verbatim that you received from the previous solution.
- If plan `focus_block` is `encoder` or `preprocessing`, also load `references/encoding_skrub.md`.
- Keep all parts listed in plan `frozen` unchanged.
- Inject `choose_*` only on the plan `focus_block`.
- Keep `choose_*` inside the DataOps `.skb.apply(...)` graph only — never assign a `choose_*` to a variable and pass it into an estimator constructor. For sklearn-API estimators use inline kwargs on `.skb.apply(Estimator(param=skrub.choose_float(...)), y=y)`; for non-sklearn estimators (e.g. CatBoost) inline kwargs do **not** resolve — use the `choose_from` variant-grid pattern (`choices_hparam_pattern.md` Pattern 4) and print the winning variant's literal params as `TUNING_BEST_PARAMS`.
- You must keep the same DataOps pipeline architecture as the structural solution; only add `choose_*` on the focus block.
- Use the same backbone estimator class(es) as the input structural solution above in your script; do not substitute a different model family.
- If the structural solution is an ensemble, tune **one** estimator leg only (the plan's focus); keep other leg(s) fixed at structural params and score with the same blend rule.
- Reproduce the structural pipeline **verbatim** — every `.skb.apply_func(...)` feature step, scaler, encoder, and column-routing step must appear unchanged in your script; if the structural solution compares variants at runtime, reproduce only its winning variant.
- Run search from the final prediction DataOp (pick `n_jobs` per Search budget below): `search = pred.skb.make_randomized_search(n_iter={n_iter}, n_jobs=n_jobs, random_state=random_state, fitted=True)`
- **Must** fit search on the training fold only: `search.fit({{"data": train_part}})`
- Evaluate holdout score with `search.best_learner_.predict({{"data": valid_part}})` (after `search.fit`).
- **TUNING_BEST_PARAMS mapping:** match each `search.best_params_` **value** to plan `tunable_params[]` by kind/range; never map by `data_op__` index. For `choose_from`, map from `search.results_.iloc[0]`. See `tuning_dataops_template.md`.
- Print holdout score as: `Final Validation Performance: {{score}}`
- Do **not** load `test_df`, refit on full `train_df`, or write `submission.csv` — holdout metric only.
- This script **must** contain `make_randomized_search`, `search.fit`, `choose_*`, and the `TUNING_BEST_PARAMS` print.

# Search budget
- `n_iter={n_iter}`. Holdout only (same split as structural). No CV.
- `make_randomized_search(..., n_jobs=...)`: search `n_jobs` parallelizes **across trials**; LightGBM/CatBoost/XGBoost (and sklearn with `n_jobs`≠1) also thread **inside each fit** — stacking both can oversubscribe CPUs and blow the timeout. **Default search `n_jobs=1`** for multithreaded backbones (predictable wall-clock, not a correctness rule). Search **`n_jobs=2`** is fine when the estimator uses `n_jobs=1` or trials are very cheap. For single-threaded sklearn, **`n_jobs=2`** is reasonable; up to **`4`** only when each trial is very fast. Never search `n_jobs=-1`.
- Reduce boosted-tree `iterations`/`n_estimators` to ~1/2 of the structural value during search.
- Keep total search time under the execution timeout ({exec_time} seconds): runtime ≈ `(n_iter + 1) * per-trial fit time`.

# Response format
- Single markdown Python code block only.
- Make sure to also print best search params found as one line using: `print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))`
- Never respond with prose-only messages such as "no more output is needed"; always return runnable Python code.
- Tool calls are preparation only; finish with runnable Python code."""

TUNE_BAKE_INSTR = """# Introduction
- Bake the tuned hyperparameters into fixed-parameter DataOps code.
- Downstream agents must not re-run search or keep `choose_*` placeholders.
- This should be a simple task, because you should just replace the `choose_*` with the best params you received from the search.
- Do not change the pipeline architecture; only touch on and replace the `choose_*` with the best params.

# Structural solution (reference)
```python
{code}
```

# Tuning plan (JSON)
{tune_plan}

# Best params from search (JSON)
{best_params_json}

# Requirements
- Replace every tuned `choose_*` with literal values from best params JSON.
- Restore structural-capacity values for any params that were reduced only for the search budget (e.g. boosted-tree `iterations`/`n_estimators` back to the structural solution's value) unless they are explicitly part of the tuned params.
- **No** `choose_*`, `make_randomized_search`, or `make_grid_search` in final code.
- Keep the same DataOps architecture and frozen blocks from the plan.
- Carry over the structural solution's full pipeline verbatim (`.skb.apply_func(...)` feature steps, scalers, encoders); only the tuned literals may differ.
- Same holdout split; print `Final Validation Performance: {{holdout_score}}` from Block 1 (`skrub.var("data", train_part)` + predict on `valid_part`) using the competition metric — not from a learner fit on full `train_df`.
- Do **not** load `test_df`, refit on full `train_df`, or write `submission.csv`.

# Response format
- Single markdown Python code block only.
- Tool calls are preparation only; finish with runnable baked Python code."""
