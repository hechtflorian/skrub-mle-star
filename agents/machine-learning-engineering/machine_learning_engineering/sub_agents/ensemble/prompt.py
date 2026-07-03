"""Defines the prompts for the ensemble agent."""

INIT_ENSEMBLE_PLAN_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- We will now provide {num_solutions} Python Solutions used for the competiton.
- Your task is to propose a plan to ensemble the {num_solutions} solutions to achieve the best performance.

{python_solutions}

# Your task
- Suggest a plan to ensemble the {num_solutions} solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- The suggested plan should be novel, effective, and easy to implement.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- Keep each solution's skrub DataOps pipeline structure intact; treat ensembling as a merge layer on top of existing predictions/models rather than rewriting preprocessing/modeling flows.
- Solutions are already terminal-tuned with fixed parameters; do not add `choose_*` to re-run hyperparameter search.
- Prefer lightweight merge plans; avoid multi-fold, multi-seed, or repeated full retrains in large spaces unless the gain clearly justifies the extra runtime or the model is small/fast.
- Load `references/ensemble_dataops_patterns.md` via skill tools; plan for Pattern A (multi-leg + blend), keeping each solution's pred chains intact.

# Respone format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since exeuction error can occur."""

ENSEMBLE_PLAN_IMPLEMENT_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you need to ensemble {num_solutions} Python Solutions for better performance based on the ensemble plan.
- We will now provide the Python Solutions and the ensemble plan.

{python_solutions}

# Ensemble Plan
{plan}

# Your task
- Implement the ensemble plan with the provided solutions.
- Unless mentioned in the ensemble plan, do not modify the origianl Python Solutions too much.
- All the provided data is already prepared and available in the `./input` directory. There is no need to unzip any files.
- The code should implement the proposed solution and print the value of the evaluation metric computed on a hold-out validation set (same split as upstream).
- Keep the original skrub DataOps pipeline(s) intact while implementing the ensemble logic. Do not refactor core preprocessing/modeling flows unless the ensemble plan explicitly requires a minimal compatibility fix.
- Input pipelines are pre-tuned fixed-parameter code; do not add `choose_*` to re-run hyperparameter search.
- Load `references/ensemble_dataops_patterns.md` via skill tools before coding; follow Pattern A skeleton, adapt legs/blend from the plan and input solutions.

# Response format required
- Your response should be a single markdown code block (wrapped in ```) which is the ensemble of {num_solutions} Python Solutions.
- There should be no additional headings or text in your response.
- Do not modify original Python Solutions especially the submission part due to formatting issue of submission.csv.
- Do not subsample or introduce dummy variables. You have to provide full new Python Solution using the {num_solutions} provided solutions.
- Print out or return a final performance metric in your answer in a clear format with the exact words: 'Final Validation Performance: {{final_validation_score}}'.
- The code should be a single-file Python program that is self-contained and can be executed as-is.
- Do not modify the original codes too much and implement the plan since new errors can occur.
- Tool calls are preparation only; you must finish by returning the final executable Python code in the same turn."""

ENSEMBLE_PLAN_REFINE_INSTR = """# Introduction
- You are a Kaggle grandmaster attending a competition.
- In order to win this competition, you have to ensemble {num_solutions} Python Solutions for better performance.
- We will provide the Python Solutions and the ensemble plans you have tried.

{python_solutions}

# Ensemble plans you have tried

{prev_plans_and_scores}

# Your task
- Suggest a better plan to ensemble the {num_solutions} solutions. You should concentrate how to merge, not the other parts like hyperparameters.
- The suggested plan must be easy to implement, novel, and effective.
- The suggested plan should be differ from the previous plans you have tried and should receive a {criteria} score.
- Keep each solution's skrub DataOps pipeline structure intact; propose refinements in ensembling logic rather than rewriting preprocessing/modeling internals.
- Prefer simple, effective ensemble variants over too heavy CV/seed grids when refining plans; only escalate compute when a prior plan showed a meaningful quality gap or the model is small/fast.
- Load `references/ensemble_dataops_patterns.md` via skill tools; plan for Pattern A (multi-leg + blend), keeping each solution's pred chains intact.

# Response format
- Your response should be an outline/sketch of your proposed solution in natural language.
- There should be no additional headings or text in your response.
- Plan should not modify the original solutions too much since exeuction error can occur."""
