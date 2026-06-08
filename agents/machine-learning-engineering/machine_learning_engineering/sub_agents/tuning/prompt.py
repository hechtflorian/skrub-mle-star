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
- Model focus: at most **3** `choose_*` nodes with tight ranges around current literals.
- Encoder/preprocessing focus: at most **2** `choose_*` nodes.
- Do not propose new feature engineering, backbone swap, or multiple focus blocks.
- Use the same holdout split as the current solution (`train_test_split` size and `random_state`).
- Search budget: `n_iter={n_iter}` (holdout randomized search only, no CV, no Optuna).

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
- Run **real** holdout randomized search with in-graph `choose_*` nodes, then report best holdout RMSE.
- Follow `references/choices_hparam_pattern.md` for the terminal tuning search contract.

# Structural solution
```python
{code}
```

# Tuning plan (JSON)
{tune_plan}

# Search budget
- `n_iter={n_iter}`, `n_jobs={n_jobs}`, `random_state=42`
- Holdout only (same split as current solution). No CV. No Optuna.

# Requirements
- Load `references/choices_hparam_pattern.md` via skill tools before editing.
- Keep all parts listed in plan `frozen` unchanged.
- Inject `choose_*` only on the plan `focus_block`.
- Keep the same DataOps architecture as the structural solution; only add `choose_*` on the focus block.
- Run search from the final prediction DataOp:
  `search = pred.skb.make_randomized_search(n_iter={n_iter}, n_jobs={n_jobs}, random_state=42, fitted=True)`
- **Must** fit search on the training fold only: `search.fit({{"data": train_part}})`
- Evaluate holdout RMSE with `search.best_learner_.predict({{"data": valid_part}})` (after `search.fit`).
- Build a JSON-serializable `best_params` dict (native Python floats/ints, not numpy scalars).
- Print holdout RMSE as: `Final Validation Performance: {{score}}`
- Print best params as one line using: `print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))`
- Keep test prediction + `submission.csv` export like the structural solution.
- This script **must** contain `make_randomized_search`, `search.fit`, `choose_*`, and the `TUNING_BEST_PARAMS` print.

# Response format
- Single markdown Python code block only.
- Never respond with prose-only messages such as "no more output is needed"; always return runnable Python code.
- Tool calls are preparation only; finish with runnable Python code."""

TUNE_BAKE_INSTR = """# Introduction
- Bake the tuned hyperparameters into fixed-parameter DataOps code.
- Downstream agents must not re-run search or keep `choose_*` placeholders.
- This should be a simple task, because you should just replace the `choose_*` with the best params you received from the search.

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
- **No** `choose_*`, `make_randomized_search`, or `make_grid_search` in final code.
- Keep the same DataOps architecture and frozen blocks from the plan.
- Same holdout split; print `Final Validation Performance: {{holdout_rmse}}` from direct holdout RMSE.
- Keep test prediction + `submission.csv` export.

# Response format
- Single markdown Python code block only.
- Tool calls are preparation only; finish with runnable baked Python code."""
