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

# Your task
- Please revise the code to fix the error.
- If the error is a 'module not found` error, then install the necessary module. You can use `pip install <module>`, where `<module>` is the name of the module to install.
- Do not remove subsampling if exists.
- Skrub DataOps skill usage: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before major edits.
- If DataOps API usage is uncertain, make focused `load_skill_resource` calls first.
- Preserve the existing DataOps pipeline structure while fixing errors (`skrub.var`/`skrub.X`/`skrub.y` / `.skb.mark_as_X()`/`.skb.mark_as_y()` + `.skb.apply(...)`); correct API usage but do not rewrite the main workflow as sklearn-only orchestration.
- If skrub DataOps usage uncertainty remains, use `site:skrub-data.org` web search and patch only the uncertain call.
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in \"./input\" directory.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""