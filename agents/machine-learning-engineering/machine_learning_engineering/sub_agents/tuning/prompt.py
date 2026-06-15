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
- Pick **one** focus block only: `model`, `encoder`, or `preprocessing`.
- Prefer `model` when ablation showed capacity or model-side effects; prefer `encoder` when encoding ablation clearly mattered.
- Model focus: at most **2** `choose_*` nodes with tight ranges around current literals.
- Encoder/preprocessing focus: at most **2** `choose_*` nodes.
- If the backbone estimator is **not** a sklearn-API estimator (`sklearn.base.BaseEstimator` subclass — e.g. CatBoost is not), numeric `choose_*` in its constructor will not resolve; plan a small discrete variant grid via `choose_from` instead (see `choices_hparam_pattern.md` Pattern 4), still with `default` values per param.
- Do not propose new feature engineering, backbone swap, or multiple focus blocks.
- Use the same holdout split as the current solution (`train_test_split` size and `random_state`).
- Search budget: `n_iter={n_iter}` (holdout randomized search only, no CV, no Optuna).
- The whole search must finish well under the 600s execution timeout: account for single-fit runtime, and plan **reduced capacity during search** for boosted trees.

# Requirements
- Call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and load `references/choices_hparam_pattern.md` via `load_skill_resource`.
- List which pipeline parts stay frozen in `frozen`.

# Response format
- Return a single JSON object only (no markdown fences, no extra text).

Use this JSON schema:
TunePlan = {{
  "focus_block": "model" | "encoder" | "preprocessing",
  "rationale": str,
  "tunable_params": [{{"name": str, "kind": "choose_from" | "choose_int" | "choose_float", ...}}],
  "frozen": [str],
  "n_iter": int
}}
Return: TunePlan"""

TUNE_IMPLEMENT_INSTR = """# Introduction
- Implement the terminal tuning plan on the structural solution below.
- Run **real** holdout randomized search with in-graph `choose_*` nodes, then report the best holdout validation score.
- Follow `references/choices_hparam_pattern.md` for the tuning search contract.

# Structural solution
```python
{code}
```

# Tuning plan (JSON)
{tune_plan}

# Search budget
- `n_iter={n_iter}`, `n_jobs={n_jobs}`, `random_state=42`
- Holdout only (same split as current solution). No CV. No Optuna.
- The script must finish well under the 600s execution timeout: reduce boosted-tree `iterations`/`n_estimators` to ~1/4 of the structural value during search, and use `n_jobs=1` if the estimator is internally multithreaded (CatBoost/LightGBM/XGBoost).

# Requirements
- Load `references/choices_hparam_pattern.md` via skill tools before editing.
- Keep all parts listed in plan `frozen` unchanged.
- Inject `choose_*` only on the plan `focus_block`.
- Keep `choose_*` inside the DataOps `.skb.apply(...)` graph only — never assign a `choose_*` to a variable and pass it into an estimator constructor. For sklearn-API estimators use inline kwargs on `.skb.apply(Estimator(param=skrub.choose_float(...)), y=y)`; for non-sklearn estimators (e.g. CatBoost) inline kwargs do **not** resolve — use the `choose_from` variant-grid pattern (`choices_hparam_pattern.md` Pattern 4) and print the winning variant's literal params as `TUNING_BEST_PARAMS`.
- You must keep the same DataOps pipeline architecture as the structural solution; only add `choose_*` on the focus block.
- Reproduce the structural pipeline **verbatim** — every `@skrub.deferred` feature function, `.skb.apply_func(...)` step, scaler, encoder, and column-routing step must appear unchanged in your script; If the structural solution compares variants at runtime, reproduce only its winning variant.
- Run search from the final prediction DataOp: `search = pred.skb.make_randomized_search(n_iter={n_iter}, n_jobs={n_jobs}, random_state=42, fitted=True)`
- **Must** fit search on the training fold only: `search.fit({{"data": train_part}})`
- Evaluate holdout score with `search.best_learner_.predict({{"data": valid_part}})` (after `search.fit`).
- Build a JSON-serializable `best_params` dict keyed by plan `tunable_params[].name` (native Python floats/ints, not numpy scalars). Map each value from `search.best_params_` to the matching plan param by kind/range — do **not** forward raw `data_op__N` keys.
- Print holdout score as: `Final Validation Performance: {{score}}`
- Print best params as one line using: `print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))`
- Do **not** load `test_df`, refit on full `train_df`, or write `submission.csv` — holdout metric only (submission stage agent adds test export later).
- This script **must** contain `make_randomized_search`, `search.fit`, `choose_*`, and the `TUNING_BEST_PARAMS` print.

# Response format
- Single markdown Python code block only.
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
- Carry over the structural solution's full pipeline verbatim (deferred feature functions, `.skb.apply_func(...)` steps, scalers, encoders); only the tuned literals may differ.
- Same holdout split; print `Final Validation Performance: {{holdout_score}}` from Block 1 (`skrub.var("data", train_part)` + predict on `valid_part`) using the competition metric — not from a learner fit on full `train_df`.
- Do **not** load `test_df`, refit on full `train_df`, or write `submission.csv`.

# Response format
- Single markdown Python code block only.
- Tool calls are preparation only; finish with runnable baked Python code."""
