"""Defines the prompts for the submission agent."""

ADD_TEST_FINAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- Input: task description + Python solution with holdout metric (Block 1).
- Your job: append Block 2 for test export (if not present already) — do not rewrite Block 1.

# Task description
{task_description}

# Python solution (Block 1 — keep unchanged)
```python
{code}
```

# Your task
- Load the test samples and create a submission file.
- Keep Block 1 exactly as given; append Block 2 after `Final Validation Performance`.
- Block 2: load `test_df` (if not present), refit on full `train_df` with the **exact same** pipeline graph, predict test, write `./final/submission.csv`.
- If ensemble: rebuild **every leg** on `train_df`; reuse the **exact same** blend/meta/threshold helpers from Block 1.
- Test data lives in `./input/`; create `./final/` if needed. 
- Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory.
- Predict every test row; do not drop samples.
- You can and should use the full training set for the final model test prediction!


# Required
- You must load `references/submission_export.md` via `list_skills` -> `load_skill`(`skrub-dataops-pipeline`) -> `load_skill_resource` before editing.
- Follow its Pattern 1 (single) or Pattern 2 (ensemble) — mirror Block 1 pipeline, only rebind `train_part` → `train_df` in Block 2.
- Do not simplify multi-leg ensemble to one model.
- If Block 2 already exists and is correct, return the script unchanged.
- Do not modify the given Python solution code too much. Try to integarte test submission with minimal changes.
- Keep the existing skrub DataOps pipeline structure intact in the final script; submission changes should focus on full-train fitting and test prediction/export, not preprocessing/modeling rewrites.
- Do not duplicate imports, defs, or Block 1; no second holdout print.
- Tool calls are prep only — finish with runnable code in the same turn.
- Do not use `exit()`; no try/except or if else to ignore unintended behavior.
- Test predictions from model(s) trained on full `train_df`; export to `./final/submission.csv`.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Do not forget the ./final/submission.csv file."""
