# Skrub Refinement Stage-1 Validation

Scope: validate the new refinement-only strategy (ablation-first + conditional bounded tuning) and decide how later stages should treat tuning.

## What was executed

- **New live run** with updated refinement prompts/skill wording:
  - archived at `submissions/california-housing-prices/gpt-5.4-mini/refinement-strategy-redesign/r1/`
- **Two comparison refinement runs** from prior branch artifacts:
  - `.../refinement-fixed-with-skill-call-ignore/v5/`
  - `.../refinement-fixed-with-skill-call-ignore/v4/`

Note: additional live runs were blocked by denied full-network permission for model calls in this environment.

## Metrics collected

Collected fields:
- search-call footprint inside refinement Python artifacts (`ablation_*.py`, `train*_improve*.py`)
- refinement score delta (`train_code_exec_result_0_1.score` -> `train_code_exec_result_1_1.score`) when `final_state.json` exists
- presence of plan-implement result keys for initial + refine implement
- live run terminal event counts for `plan_implement_initial_agent_1` and `plan_implement_agent_1`

### Results

| run | search calls in refinement files | baseline score | refined score | delta (lower is better) | implement evidence |
|---|---:|---:|---:|---:|---|
| `r1_new` | 0 | N/A | N/A | N/A | terminal events: initial=7, refine=9 |
| `v5_baseline` | 0 | 56026.0863 | 55963.4561 | +62.6302 improvement | exec keys for both initial/refine implement present |
| `v4_baseline` | 4 | 57402.2692 | 57402.2692 | +0.0000 | exec keys for both initial/refine implement present |

Interpretation:
- A run with **no refinement-stage search** can still improve (`v5`).
- A run with **higher search footprint** did **not** improve (`v4`).
- This supports the policy to make search conditional in refinement instead of default.

## Observed blocker in new live run

- Pipeline crashed later in submission stage with:
  - `KeyError: 'final_validation_score'` while formatting submission prompt text.
- This is outside refinement policy itself, but it prevents full end-to-end completion in the current branch.

## Decision for later stages

For now, **ensemble/submission should inherit reuse-first policy and avoid additional tuning by default**:
- reuse strongest known params from refinement outputs
- only tune later if there is explicit score regression or unresolved untuned placeholders tied to changed search-sensitive components
- keep full/expensive search for a dedicated final optimization pass (not in every ensemble/submission iteration)

This keeps compute controlled and aligns with the new refinement strategy.
