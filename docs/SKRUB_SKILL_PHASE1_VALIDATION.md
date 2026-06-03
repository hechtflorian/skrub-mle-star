# skrub Skill Phase-1 Validation

## Scope
- Objective: compare baseline behavior vs stricter skrub-oriented setup and estimate skill context overhead.
- Current limitation: no same-model/same-seed run with the new skill tooling has been executed yet in this session.
- Method used now: artifact-based comparison from existing run outputs plus a reusable checker script.

## Compared artifacts
- A (`prompt-only`): `submissions/california-housing-prices/gpt-5.4-mini-skrub-prompt-only/final_state.json`
- B (`prompt-hardened reference`): `submissions/california-housing-prices/mistral/run-3-skrub-prompt-init/final_state.json`
- Script: `scripts/eval_dataops_adherence.py`

## Results
- **A_prompt_only_gpt5**
  - model: `openai/gpt-5.4-mini`
  - score (lower is better): `56096.43152837484`
  - retry proxy (`*bug_summary*` entries): `2`
  - DataOps anchor hits: `4 / 8`
  - sklearn-pipeline pattern detected: `true`
- **B_prompt_hardened_mistral**
  - model: `openai/mistral-large-3-675b-instruct-2512`
  - score (lower is better): `1000000000.0` (failure-like sentinel)
  - retry proxy (`*bug_summary*` entries): `0`
  - DataOps anchor hits: `0 / 8`
  - sklearn-pipeline pattern detected: `true`

## Token/context overhead estimate for skill package
- L2 (`SKILL.md`): ~`375.2` tokens (rough estimate: chars/4)
- L3 references bundle (all markdown files): ~`1249.5` tokens
- Practical implication: keep L3 loading selective (`load_skill_resource`) to avoid unnecessary context cost.

## Decision for phase 2 rollout
- Do **not** expand to refinement/ensemble/submission yet.
- Keep rollout at phase-1 attachment points (initialization retriever + debug agents) until a controlled same-model A/B run is captured with the new skill-enabled code path.

## Next controlled validation command
Run this once a new skill-enabled run artifact is available:

```bash
uv run --project "mle-star_improved/agents/machine-learning-engineering" \
  python "mle-star_improved/scripts/eval_dataops_adherence.py" \
  --run-a "<prompt_only_final_state.json>" \
  --label-a "A_prompt_only" \
  --run-b "<skill_enabled_final_state.json>" \
  --label-b "B_skill_enabled" \
  --skill-md "mle-star_improved/agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/SKILL.md" \
  --skill-ref-dir "mle-star_improved/agents/machine-learning-engineering/machine_learning_engineering/skills/skrub-dataops-pipeline/references"
```
