"""Defines the prompts for the refinement agent."""

ABLATION_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.

# Python solution
```python
{code}
```
# Instructions
- You need to generate a simple Python code that performs an ablation study on the above Python solution script.
- The generated code should create variations by modifying or disabling parts (1-2 simple parts) of the training process.
- For each ablation, print out how the modification affects the model's performance.

# Requirements
- You must call `list_skills` -> `load_skill` -> `load_skill_resource` for `skrub-dataops-pipeline` before editing.
- Use focused `load_skill_resource` reference calls for uncertain skrub API details.
- You must call at least one of the following references with `load_skill_resource` before editing: `references/choices_hparam_pattern.md` for DataOps/choice hyperparameter tuning; `references/dataops_tuning_optuna.md` when the plan explicitly uses Optuna.
- If you intend to report a variant as tuned, run real search (`.skb.make_randomized_search(...)`, `.skb.make_grid_search(...)`, or Optuna trial flow) and evaluate the searched model.
- For quick ablation checks or minor non-hparam changes, prefer fixed values or previously strong params instead of rerunning full search.
- Do not treat `.skb.make_learner(...)` / `.skb.eval(...)` default-choice behavior as tuned results.
- Tool calls are preparation only; you must finish by returning executable Python code for the ablation study in the same turn.

# Response format
- There should be no additional headings or text in your response.
- The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.
- The code should include a printing statement that shows the performance of each ablation.
- The code should consequently print out which part of the code contributes the most to the overall performance.
- Return Python code; never return only tool-call results.
"""

ABLATION_SEQ_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.
- We will also provide the summaries of previous ablation studies.

# Python solution
```python
{code}
```

{prev_ablations}

# Instructions
- You need you to generate a simple Python code that performs an ablation study on the train.py script.
- The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.
- Your ablation study should concentrate on the other parts that have not been previously considered.
- For each ablation, print out how the modification affects the model's performance.

# Requirements
- You must call `list_skills` -> `load_skill` -> `load_skill_resource` for `skrub-dataops-pipeline` before editing.
- Use focused `load_skill_resource` reference calls for uncertain skrub API details.
- You must call at least one of the following references with `load_skill_resource` before editing: `references/choices_hparam_pattern.md` for DataOps/choice hyperparameter tuning; `references/dataops_tuning_optuna.md` when the plan explicitly uses Optuna.
- If you intend to report a variant as tuned, run real search (`.skb.make_randomized_search(...)`, `.skb.make_grid_search(...)`, or Optuna trial flow) and evaluate the searched model.
- For quick ablation checks or minor non-hparam changes, prefer fixed values or previously strong params instead of rerunning full search.
- Do not treat `.skb.make_learner(...)` / `.skb.eval(...)` default-choice behavior as tuned results.
- Tool calls are preparation only; you must finish by returning executable Python code for the ablation study in the same turn.

# Response format
- There should be no additional headings or text in your response.
- The Python code for the ablation study should not load test data. It should only focus on training and evaluating the model on the validation set.
- The code should include a printing statement that shows the performance of each ablation.
- The code should consequently print out what part of the code contributes the most to the overall performance.
- Return Python code; never return only tool-call results.
"""

SUMMARIZE_ABLATION_INSTR = """# Your code for ablation study was:
```python
{code}
```

# Ablation study results after running the above code:
{result}

# Your task
- Summarize the result of ablation study based on the code and printed output.
"""

EXTRACT_BLOCK_AND_PLAN_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to extract a code block from the current Python solution and improve the extracted block for better performance.
- Your suggestion should be based on the ablation study results of the current Python solution.
- We will now provide the current Python solution and the ablation study results.

# Python solution
```python
{code}
```

# Ablation study results
{ablation_results}

# Your task
- Given the ablation study results, suggest an effective next plan to improve the above Python script.
- The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- Prefer plans that improve high-impact DataOps graph parts first (table assembly, `.skb.apply(...)` learner path, and `choose_*` choices) rather than replacing the architecture.
- Also extract the code block from the above Python script that need to be improved according to the proposed plan.

# Requirements
- You must call `list_skills` -> `load_skill` -> `load_skill_resource` for `skrub-dataops-pipeline` before finalizing the plan.
- Use focused `load_skill_resource` reference calls for uncertain skrub API details; preserve DataOps architecture.
- You must call at least one of the following references with `load_skill_resource` before editing: `references/choices_hparam_pattern.md` for DataOps/choice hyperparameter tuning; `references/dataops_tuning_optuna.md` when the plan explicitly uses Optuna.
- If your proposed plan claims tuning with `choose_*` / `choose_from(...)`, include real search execution and best-result model selection.
- If changes are minor and search-sensitive parts are unchanged, prefer reusing previously strong params or explicit fixed values.
- Do not describe default-choice execution as tuned.
- Tool calls are preparation only; you must finish by returning your proposed solution.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.

Use this JSON schema:

Refine_Plan = {{'code_block': str, 'plan': str}}
Return: list[Refine_Plan]"""

EXTRACT_BLOCK_AND_PLAN_SEQ_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to extract a code block from the current Python solution and improve the extracted block for better performance.
- Your suggestion should be based on the ablation study results of the current Python solution.
- We will now provide the current Python solution and the ablation study results.
- We also provide code blocks which you have tried to improve previously.

# Python solution
```python
{code}
```

# Ablation study results
{ablation_results}

{prev_code_blocks}

# Your task
- Given the ablation study results, suggest an effective next plan to improve the above Python script.
- The plan should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- Please avoid plan which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- Try to improve the other part which was not considered before.
- Prefer plans that improve untried high-impact DataOps graph parts first (table assembly, `.skb.apply(...)` learner path, and `choose_*` choices).
- Also extract the code block from the above Python script that need to be improved according to the proposed plan. You should try to extract the code block which was not improved before.

# Requirements
- You must call `list_skills` -> `load_skill` -> `load_skill_resource` for `skrub-dataops-pipeline` before finalizing the plan.
- Use focused `load_skill_resource` reference calls for uncertain skrub API details; preserve DataOps architecture.
- You must call at least one of the following references with `load_skill_resource` before editing: `references/choices_hparam_pattern.md` for DataOps/choice hyperparameter tuning; `references/dataops_tuning_optuna.md` when the plan explicitly uses Optuna.
- If your proposed plan claims tuning with `choose_*` / `choose_from(...)`, include real search execution and best-result model selection.
- If changes are minor and search-sensitive parts are unchanged, prefer reusing previously strong params or explicit fixed values.
- Do not describe default-choice execution as tuned.
- Tool calls are preparation only; you must finish by returning your proposed solution.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.

Use this JSON schema:

Refine_Plan = {{'code_block': str, 'plan': str}}
Return: list[Refine_Plan]"""

PLAN_REFINEMENT_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to improve the code block for better performance.
- We will provide the code block you are improving and the improvement plans you have tried.

# Code block
```python
{code_block}
```

# Improvement plans you have tried

{prev_plan_summary}

# Your task
- Suggest a better plan to improve the above code block.
- The suggested plan must be novel and effective.
- Please avoid plans which can make the solution's running time too long (e.g., searching hyperparameters in a very large search space).
- The suggested plan should be differ from the previous plans you have tried and should receive a higher score.

# Requirements
- You must call `list_skills` -> `load_skill` -> `load_skill_resource` for `skrub-dataops-pipeline` before finalizing the plan.
- Use focused `load_skill_resource` reference calls for uncertain skrub API details; preserve DataOps architecture.
- You must call both of the following references with `load_skill_resource` before finalizing the plan: `references/choices_hparam_pattern.md` and`references/dataops_tuning_optuna.md`.
- If the plan claims tuned `choose_*` / `choose_from(...)`, it must include real search execution and best-result selection.
- Prefer reuse/fixed params when only minor non-hparam edits are proposed; avoid redundant full searches.
- Do not call a plan tuned unless search is executed; default-choice learners are baseline behavior only.
- Tool calls are preparation only; you must finish by returning your proposed solution plan.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- There should be no additional headings or text in your response.
"""

IMPLEMENT_PLAN_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need refine the code block for better performance based on the improvement plan.
- We will now provide the code block and the improvement plan.

# Code block
```python
{code_block}
```

# Improvement plan
{plan}

# Your task
- Implement the improvement plan on the above code block. But do not remove subsampling if exists.
- The code block should be improved according to the proposed plan.
- Note that all the variable including actual data is defined earlier (since you are just seeing a code block), therefore do not introduce dummy variables.
- Keep the refined block compatible with the existing DataOps pipeline structure (`skrub.var`/`skrub.X`/`skrub.y` / `.skb.mark_as_X()`/`.skb.mark_as_y()` + `.skb.apply(...)`).
- Do not convert or replace the main pipeline with sklearn-only `Pipeline`/`ColumnTransformer` orchestration.

# Requirements
- For skrub DataOps, you must load `list_skills` -> `load_skill` -> `load_skill_resource` for `skrub-dataops-pipeline` first.
- Use `load_skill_resource` calls with the most relevant references for uncertain parts before editing.
- You must call at least one of the following references with `load_skill_resource` before editing: `references/choices_hparam_pattern.md` for DataOps/choice hyperparameter tuning; `references/dataops_tuning_optuna.md` when the plan explicitly uses Optuna.
- If the improved block is intended to be tuned with `choose_*` / `choose_from(...)`, implement real search (`.skb.make_randomized_search(...)`, `.skb.make_grid_search(...)`, or Optuna trial flow) and use the best result in training/inference.
- If only minor non-hparam changes are made, prefer reusing previous strong params or fixed values instead of rerunning full search.
- Do not label default-choice execution as tuned.
- Tool calls are preparation only; you must finish by returning executable Python code for the improved code block.

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the improved code block.
- There should be no additional headings or text in your response.
"""
