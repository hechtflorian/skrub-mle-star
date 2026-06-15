"""Defines the prompts for the submission agent."""

ADD_TEST_FINAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to come up with an excellent solution in Python.
- We will now provide a task description and a Python solution.
- What you have to do on the solution is just loading test samples and create a submission file.

# Task description
{task_description}

# Python solution
```python
{code}
```

# Your task
- Load the test samples and create a submission file.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Test data is available in the `./input` directory.
- Save the test predictions in a `submission.csv` file. Put the `submission.csv` into `./final` directory.
- You should not drop any test samples. Predict the target value for all test samples.
- This is a very easy task because the only thing to do is to load test samples and then replace the validation samples with the test samples. Then you should use the full training set for the final model!

# Required
- You must load `references/dataops_api_quickmap.md` via `list_skills` -> `load_skill`(`skrub-dataops-pipeline`) and use `load_skill_resource() to load `references/dataops_api_quickmap.md` before editing (use the **Submission stage only** section for full-train refit, `test_df` predict, and export).
- Do not modify the given Python solution code too much. Try to integarte test submission with minimal changes.
- Keep the existing skrub DataOps pipeline structure intact in the final script; submission changes should focus on full-train fitting and test prediction/export, not preprocessing/modeling rewrites.
- The pipeline is pre-tuned with fixed parameters; do not add `choose_*` or re-run hyperparameter search.
- Tool calls are prep only — finish with runnable code in the same turn.
- There should be no additional headings or text in your response.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not forget the ./final/submission.csv file.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
- The final test predictions must be generated from model(s) trained on the full training set.
- Before fitting on the full training set and writing `./final/submission.csv`, keep and run the solution's existing holdout validation block unchanged (same `train_test_split` size and `random_state`). Print only that holdout validation score as `Final Validation Performance: {{final_validation_score}}`. Then refit on the full training set for final test predictions."""
