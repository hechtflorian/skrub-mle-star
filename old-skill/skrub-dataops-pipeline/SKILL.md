---
name: skrub-dataops-pipeline
description: Central hub for building and debugging end-to-end ML pipelines with strict usage of skrub DataOps abstractions without falling back to sklearn-only orchestration. Trigger this skill when the user prompts you to use skrub DataOps in tasks involving model retrieval, code generation, or debugging for ML pipelines.
---

# Skrub DataOps Pipeline
This file is the directory and execution guide for building and debugging end-to-end ML pipelines with `skrub` DataOps.
Load only the relevant reference files for the current subtask to avoid context bloat.

## Rules:
1. Build the solution as a DataOps graph first, then write code.
2. Keep the main pipeline in skrub DataOps primitives (`skrub.var` or `skrub.X`/`skrub.y`, then `.skb.apply(...)`).
3. Keep the architecture in DataOps while debugging; patch API calls, do not rewrite the workflow into sklearn-only orchestration.
4. If uncertain about an API, load the relevant reference file and only then patch the uncertain call.
5. Reject and rewrite solutions where skrub is only incidental (for example, only `TableVectorizer` in an otherwise sklearn-only orchestration).

## Core Concepts Directory
Refer to these files for foundational knowledge on skrub and skrub DataOps:
- **Skrub DataOps API overview**: [dataops_api_quickmap.md](references/dataops_api_quickmap.md)
  - DataOps primitives, `skb` methods, and a complete multi-table DataOps example.
- **General Skrub API overview**: [skrub_general_api.md](references/skrub_general_api.md)
  - Core non-DataOps API: pipeline utilities, encoders, cleaning, joining, and selectors.

## Reference examples demonstrating skrub usage
- **Example ML pipeline using skrub DataOps**: [dataops_api_quickmap.md](references/dataops_api_quickmap.md)
- **Hyperparameter tuning with skrub**: [choices_hparam_pattern.md](references/choices_hparam_pattern.md)
  - Choice-based tuning (`choose_from`, `choose_int`, `choose_float`) inside DataOps graphs.
- **Tuning DataOps objects with skrub and optuna**: [dataops_tuning_optuna.md](references/dataops_tuning_optuna.md)
  - Optuna backend workflow for tuning choices in DataOps plans.
- **Encoding tabular data with skrub**: [encoding_skrub.md](references/encoding_skrub.md)
  - `TableVectorizer` and tabular pipeline encoding patterns.
- **Joining across columns/tables**: [joining_across_columns.md](references/joining_across_columns.md)
  - Join-oriented preprocessing patterns for multi-table tasks.
- **Subsampling DataOps for faster iteration**: [skrub_subsampling.md](references/skrub_subsampling.md)
  - Safe use of `.skb.subsample()` during iterative development.
- **Debugging failure patterns**: [common_failure_fixes.md](references/common_failure_fixes.md)
  - Fast triage patterns for imports, API signatures, and graph consistency.

## Reference loading protocol
1. Start with `references/dataops_api_quickmap.md` for DataOps structure and symbol checks and `references/skrub_general_api.md` for general non-dataop skrub API.
2. Load one targeted deep reference based on the current task:
   - tuning: `references/choices_hparam_pattern.md` or `references/dataops_tuning_optuna.md`
   - encoding: `references/encoding_skrub.md`
   - joining/multi-table: `references/joining_across_columns.md`
   - fast iteration: `references/skrub_subsampling.md`
   - debugging/failures: `references/common_failure_fixes.md`
3. If uncertainty remains, use focused web search on `site:skrub-data.org` and patch only the uncertain call.

## Output contract
- Return runnable single-file Python code when code is requested.
- Keep the metric print line with exact text: `Final Validation Performance: {final_validation_score}`.
