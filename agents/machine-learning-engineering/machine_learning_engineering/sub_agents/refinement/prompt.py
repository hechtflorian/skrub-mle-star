"""Defines the prompts for the refinement agent."""

ABLATION_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to perform an ablation study on the current Python solution to know which parts of the code contribute the most to the overall performance.
- We will now provide a current Python solution.

# Python solution
```python
{code}
```

# Data profile (precomputed on train.csv)
{data_profile}

# Instructions
- You need to generate a simple Python code that performs an ablation study on the above Python solution script.
- The generated code should create variations by modifying or disabling parts (1-2 simple parts) of the training process.
- For each ablation, print out how the modification affects the model's performance.

# Requirements
- You must call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before editing and load relevant references via `load_skill_resource` if you see fit. 
- If an ablation variant changes feature encoding/preprocessing/column routing, load `references/encoding_skrub.md` and/or `references/selectors_routing_skrub.md` via `load_skill_resource` before finalizing code.
- If the data profile suggests redundant columns, ratios, cleaning, or derived features, load `references/feature_engineering_skrub.md` via `load_skill_resource` before finalizing code.
- If you want to perform parameter search (e.g. model family, encoders, specific values), load `references/choices_hparam_pattern.md` via `load_skill_resource` before finalizing code.
- Use the data profile to choose focused ablations if useful.
- Keep ablation search budget small when used (`n_iter <= 5`).
- Keep the same model family and validation split as the current solution; unless changing the model is the explicit hypothesis.
- If the profile shows string/categorical complexity or missingness, include at least one preprocessing/encoding structural ablation.
- If the profile shows highly correlated feature pairs, include at least one redundancy/ratio/drop-one ablation.
- Do not claim tuning from default-choice `.skb.make_learner(...)` / `.skb.eval(...)` behavior.
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

# Data profile (precomputed on train.csv)
{data_profile}

# Instructions
- You need to generate simple Python code that performs an ablation study on the train.py script.
- The generated code should create variations by modifying or disabling parts (2-3 parts) of the training process.
- Your ablation study should concentrate on the other parts that have not been previously considered.
- For each ablation, print out how the modification affects the model's performance.

# Requirements
- You must call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before editing and load relevant references via `load_skill_resource` if you see fit.
- If an ablation variant changes feature encoding/preprocessing/column routing, load `references/encoding_skrub.md` and/or `references/selectors_routing_skrub.md` via `load_skill_resource` before finalizing code.
- If the data profile suggests redundant columns, ratios, cleaning, or derived features, load `references/feature_engineering_skrub.md` via `load_skill_resource` before finalizing code.
- If you want to perform parameter search (e.g. model family, encoders, specific values), load `references/choices_hparam_pattern.md` via `load_skill_resource` before finalizing code.
- Keep ablation search budget small when used (`n_iter <= 5`).
- Use the data profile to choose focused ablations if useful.
- Keep the same model family and validation split as the current solution; unless changing the model is the explicit hypothesis.
- If the profile shows string/categorical complexity or missingness, include at least one preprocessing/encoding structural ablation.
- If the profile shows highly correlated feature pairs, include at least one redundancy/ratio/drop-one ablation.
- Do not claim tuning from default-choice `.skb.make_learner(...)` / `.skb.eval(...)` behavior.
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
- Also extract the code block from the above Python script that needs to be improved according to the proposed plan.

# Requirements
- You must call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` before finalizing the plan and load relevant references via `load_skill_resource` if you see fit.
- If your plan changes feature encoding/preprocessing or column routing, load `references/encoding_skrub.md` and/or `references/selectors_routing_skrub.md` via `load_skill_resource` before finalizing the plan.
- If your plan adds derived features, drops redundant columns, or applies cleaning/scaling from the data profile, load `references/feature_engineering_skrub.md` via `load_skill_resource` before finalizing the plan.
- If you want to perform hyperparameter search (e.g. over model family, encoding, or specific values), load `references/choices_hparam_pattern.md` via `load_skill_resource` before finalizing.
- If tuning is proposed, state a focused bounded search budget (`n_iter <= 8`) and a compact targeted search space.
- Do not call a plan tuned unless search is actually executed in implementation.
- You must always finish by returning your proposed solution, even if you call `list_skills`/`load_skill`/`load_skill_resource`.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.
- Tool calls are preparation only; you must finish by returning your proposed solution.

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
- Also extract the code block from the above Python script that needs to be improved according to the proposed plan. You should try to extract a code block that was not improved before.

# Requirements
- You must call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and load relevant references via `load_skill_resource` before finalizing the plan.
- If your plan changes feature encoding/preprocessing or column routing, load `references/encoding_skrub.md` and/or `references/selectors_routing_skrub.md` via `load_skill_resource` before finalizing the plan.
- If your plan adds derived features, drops redundant columns, or applies cleaning/scaling from the data profile, load `references/feature_engineering_skrub.md` via `load_skill_resource` before finalizing the plan.
- If you want to perform hyperparameter search (e.g. over model family, encoding, or specific values), load `references/choices_hparam_pattern.md` via `load_skill_resource` before finalizing the plan.
- If tuning is proposed, state a focused bounded search budget (`n_iter <= 8`) and a compact targeted search space.
- Do not call a plan tuned unless search is actually executed in implementation.
- You must always finish by returning your proposed solution, even if you call `list_skills`/`load_skill`/`load_skill_resource`.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences) and a single markdown code block which is the code block that need to be improved.
- The code block can be long but should be exactly extracted from the Python script provided above.
- Tool calls are preparation only; you must finish by returning your proposed solution.

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
- The suggested plan should differ from the previous plans you have tried and should receive a higher score.

# Requirements
- You must call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and load relevant references via `load_skill_resource`before finalizing the plan.
- If your plan changes feature encoding/preprocessing or column routing, load `references/encoding_skrub.md` and/or `references/selectors_routing_skrub.md` via `load_skill_resource` before finalizing the plan.
- If your plan adds derived features, drops redundant columns, or applies cleaning/scaling from the data profile, load `references/feature_engineering_skrub.md` via `load_skill_resource` before finalizing the plan.
- If you want to perform parameter search (e.g. model family, encoders, specific values), load `references/choices_hparam_pattern.md` via `load_skill_resource` before finalizing the plan.
- Prefer fixed/reused params when edits are minor and search-sensitive parts are unchanged.
- If proposing tuning, keep it bounded (`n_iter <= 8`) with a compact targeted search space.
- Do not describe default-choice `.skb.make_learner(...)` / `.skb.eval(...)` runs as tuned.
- You must always finish by returning your proposed solution, even if you call `list_skills`/`load_skill`/`load_skill_resource`.

# Response format
- Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences).
- There should be no additional headings or text in your response.
- Tool calls are preparation only; you must finish by returning your proposed solution plan.
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

# Requirements
- You must call `list_skills` -> `load_skill` for `skrub-dataops-pipeline` and load relevant references via `load_skill_resource` before editing.
- If implementation changes feature encoding/preprocessing or column routing, load `references/encoding_skrub.md` and/or `references/selectors_routing_skrub.md` via `load_skill_resource` before finalizing code.
- If implementation adds derived features, drops redundant columns, or applies cleaning/scaling, load `references/feature_engineering_skrub.md` via `load_skill_resource` before finalizing code.
- If the plan adds derived features via `.skb.apply_func(...)`, do not build column lists from raw `train_part`/`train_df`; route columns with skrub selectors on the post-FE `X` graph (or use a single `TableVectorizer()` on all features).
- If you want to perform hyperparameter search (e.g. over model family, encoding, or search-space nodes), load `references/choices_hparam_pattern.md` and `references/dataops_tuning_optuna.md` via `load_skill_resource` before finalizing code.
- If the plan is non-tuning or the change is minor, prefer reusing previous strong parameters or explicit fixed values.
- If the plan indicates hyperparameter tuning is valuable, you must run real search (`.skb.make_randomized_search(...)`, `.skb.make_grid_search(...)`, or Optuna trial flow), keep it tightly bounded (`n_iter <= 8`), and use the best result.
- Do not label default-choice `.skb.make_learner(...)` / `.skb.eval(...)` behavior as tuned.
- Keep search localized to the selected impactful block, preserve existing DataOps architecture, and keep the search space compact.
- The printed `Final Validation Performance` must be computed from a direct holdout RMSE (`mean_squared_error(y_val, y_pred) ** 0.5`) on a real validation split.
- Do not use transformed generic CV scores (e.g., `sqrt(mean(test_score))`) as the final refinement score line.
- You must always return your final runnable Python code block in the same response, even if you call tools via `list_skills`/`load_skill`/`load_skill_resource`.

# Response format
- Your response should be a single markdown code block (wrapped in ```) which is the improved code block.
- There should be no additional headings or text in your response.
- You must finish by returning executable Python code for the improved code block in this same response.
- Tool calls are preparation only; you must always finish by returning runnable Python code.
- Never return plain text such as "No more outputs are needed."; always return a runnable Python code block for this step.
"""
