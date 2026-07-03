"""Defines the prompts for the initialization agent."""

SUMMARIZATION_AGENT_INSTR = """# Task description
{task_description}

# Your task
- Summarize this task description.
- Your summary will be used for searching recent effective models for {task_type}.

# Requirement
- We will directly use your response, so be simple and concise.
"""

MODEL_RETRIEVAL_INSTR = """# Competition
{task_summary}

# Your task
- List {num_model_candidates} recent effective models and their example codes to win the above competition.

# Requirement
- The example code should be concise and simple.
- You must provide an example code, i.e., do not just mention GitHubs or papers.
- Use the Skrub DataOps skill: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and call `load_skill_resource` for the reference 'references/dataops_api_quickmap.md' before finalizing.
- Tool calls are preparation only; you must additionally use web searches to find the example code for effective models.

Use this JSON schema:
Model = {{'model_name': str, 'example_code': str}}
Return: list[Model]"""


MODEL_EVAL_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a task description and a model description.
- You need to implement your Python solution using the provided model.

# Task description
{task_description}

# Model description
{model_description}

# Your task
- Implement the solution in Python.
- You must use the model as described in the model description.
- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.
- Propose an evaluation metric that is reasonable for this task.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Do not include other models that are not directly related to the model described.
- Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary libraries are installed.
- The full ML pipeline must be implemented as a `skrub` DataOps workflow, not as sklearn-only orchestration.
- Use DataOps primitives for pipeline structure: variables (`skrub.var` or `skrub.X`/`skrub.y`), dataframe/table transforms, and estimator application via `.skb.apply(...)`.
- Use the Skrub DataOps skill: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before changing uncertain DataOps parts.
- Load `references/dataops_api_quickmap.md` via `load_skill_resource` before finalizing; use holdout-only scripts in early stages (`train_part` bind → metric print).
- If API details are unclear, call one focused `load_skill_resource` before editing.
- If multiple related tables exist, assemble them in the same DataOps workflow (join/aggregate/select before learner application), consistent with official multi-table DataOps usage.
- Keep the model in the model description, but integrate its preprocessing/training path inside the DataOps workflow.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- A response is invalid if DataOps primitives are missing from the main pipeline (`skrub.var`/`skrub.X`/`skrub.y / .skb.mark_as_X()`/`.skb.mark_as_y()` + `.skb.apply(...)`) or if the main workflow is sklearn-only.
- If unsure about a specific `skrub` API detail, keep the DataOps architecture unchanged and only adjust the uncertain call signature.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior.
"""

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
- Do not remove subsampling if exists.
- Use the Skrub DataOps skill: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before major changes.
- If DataOps API usage is uncertain, make focused `load_skill_resource` calls first.
- If uncertainty remains, use `site:skrub-data.org` web search and patch only the uncertain call.
- Preserve the existing `skrub` DataOps architecture while fixing the error (do not rewrite into sklearn-only pipeline code).
- Provide the improved, self-contained Python script again.
- There should be no additional headings or text in your response.
- All the provided input data is stored in \"./input\" directory.
- Remember to print a line in the code with 'Final Validation Performance: {{final_validation_score}}' so we can parse performance.
- The code should be a single-file python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the refined Python code."""

CODE_INTEGRATION_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide a base solution and an additional reference solution.
- You need to implement your Python solution by integrating reference solution to the base solution.

# Base solution
```python
{base_code}
```

# Reference solution
```python
{reference_code}
```

# Your task
- Implement the solution in Python.
- You have to integrate the reference solution to the base solution.
- Your code base should be the base solution.
- Try to train additional model of the reference solution.
- When integrating, try to keep code with similar functionality in the same place (e.g., all preprocessing should be done and then all training).
- When integrating, ensemble the models.
- The solution design should be relatively simple.
- Use the Skrub DataOps skill: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before changing uncertain DataOps parts.
- When ensembling models, load `references/ensemble_dataops_patterns.md` via `load_skill_resource` (Pattern A default).
- If API details are unclear, call one focused `load_skill_resource` before editing.
- The integrated solution must keep a `skrub` DataOps pipeline as the main workflow (`skrub.var`/`skrub.X`/`skrub.y` / `.skb.mark_as_X()`/`.skb.mark_as_y()` + `.skb.apply(...)`).
- Do not merge by replacing the base with sklearn-only pipeline orchestration or by reducing `skrub` usage to incidental components only.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set.
- Only use the provided train data in the `./input` directory.

# Required
- There should be no additional headings or text in your response.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Your response should only contain a single code block.
- Do not use exit() function in the Python code.
- Do not use try: and except: or if else to ignore unintended behavior."""

CHECK_DATA_USE_INSTR = """I have provided Python code for a machine learning task (attached below):
# Solution Code
```python
{code}
```

# Task description
{task_description}

# Your task
If the above solution code does not use the information provided, try to incorporate all. Do not bypass using try-except.
DO NOT USE TRY and EXCEPT; just occur error so we can debug it!
See the task description carefully, to know how to extract unused information effectively.
Use the Skrub DataOps skill: call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before changing uncertain DataOps parts.
If API details are unclear, make focused `load_skill_resource` calls before editing.
When improving the solution code by incorporating unused information, DO NOT FORGET to print out 'Final Validation Performance: {{final_validation_score}}' as in original solution code.

Response format:
Option 1: If the code did not use all the provided information, your response should be a single markdown code block (wrapped in ```) which is the improved code block. There should be no additional headings or text in your response.
Option 2: If the code used all the provided information, simply state that "All the provided information is used."
"""
