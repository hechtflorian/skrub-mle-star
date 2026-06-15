"""Defines the prompts for debugging."""

BUG_SUMMARY_INSTR = """# Error report
{bug}

# Your task
- Remove all unnecessary parts of the above error report.
- We are now running {filename}.py. Do not remove where the error occurred."""

BUG_REFINE_INSTR = """# Task description
{task_description}

# Code with an error:
{code}

# Error:
{bug}
{backbone_contract}
# Your task
- Please revise the code to fix the error.
- Make the **smallest possible change** that fixes the reported error. Do not regenerate the script from scratch, do not simplify or remove working logic (e.g. ablation variants, derived features, search calls), and do not add new stages the input code did not have (no test loading or submission writing if the input code had none).
- If the error is a 'module not found` error, then install the necessary module. You should use `pip install <module>`, where `<module>` is the name of the module to install.
- Do not remove subsampling if exists.
- Only for skrub-related errors: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and load relevant references via `load_skill_resource` before major edits (not for errors unrelated to skrub).
- Preserve the existing DataOps pipeline structure while fixing errors (`skrub.var`/`skrub.X`/`skrub.y` / `.skb.mark_as_X()`/`.skb.mark_as_y()` + `.skb.apply(...)`) and do not rewrite the main workflow as sklearn-only orchestration.
- If unsure about skrub DataOps usage to keep the pipeline structure, make focused `load_skill_resource` calls — load only the 2-3 most relevant references (start with `references/dataops_api_quickmap.md`); do not load the whole reference set unless explicitly relevant.
- When fixing validation logic, preserve honest holdout binding: metric line uses `train_part`, not full `train_df` (see holdout section in `dataops_api_quickmap.md`).
- Only if skrub DataOps usage uncertainty remains even after skill tool calls, use `site:skrub-data.org` web search and patch only the uncertain call.
- You **must** keep the same backbone estimator/model class as the input code; never swap the model family to silence an error unless the error is truly unfixable without a class change.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in \"./input\" directory.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""