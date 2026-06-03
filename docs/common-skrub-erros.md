# Common skrub errors (runtime notes)

This file tracks recurrent failures seen in generated scripts and the fixed usage
patterns that should be preferred by default.

## 1) DataOps chaining and `.fit(...)` mismatch

### Symptom
- `'Series' object has no attribute 'fit'`
- `TypeError: SkrubLearner._eval_in_mode() takes 3 positional arguments but 4 were given`

### Root cause
- A DataOps chain created with `.skb.apply(...)` is treated like a raw sklearn estimator.

### Correct pattern
```python
pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y)
cv_results = pred.skb.cross_validate()
learner = pred.skb.make_learner(fitted=True)
```

### Avoid
- `pred.fit(X, y)`
- `pred.skb.make_learner().fit(X, y)` with positional `X, y` style.

## 2) `predict(...)` environment violation

### Symptom
- `TypeError: environment should be a dictionary of input values`

### Root cause
- Passing a DataOp node or raw array directly to a compiled learner.

### Correct pattern
```python
pred_test = learner.predict({"data": test_df})
```

For multi-table plans, pass all required named sources, e.g.
`{"baskets": baskets_df, "products": products_df}`.

## 3) Proxy/type intrusion from mixing DataOps and eager pandas/sklearn

### Symptom
- Attribute errors on `.dtype` / `.to_pandas`
- crashes when wrapping lazy DataOps outputs inside external constructors too early

### Root cause
- Lazy DataOps proxies are forced through eager API paths.

### Correct pattern
- Keep preprocessing/modeling inside DataOps (`.skb.apply(...)`) until learner/search.
- Compile to learner before external prediction/evaluation calls.

## 4) Unsupported/guessed arguments

### Symptom
- `TypeError` from unsupported kwargs (example: `TableVectorizer(impute_missing_values=True)`)
- `TypeError` from unsupported subsample signature (`.skb.subsample(..., random_state=0)`)

### Correct pattern
```python
vectorizer = skrub.TableVectorizer(...)
sub_data = data.skb.subsample(n=100)
```

### Rule
- Only use signatures that exist in the local skill references.

## 4b) `choose_from` dictionary key type error

### Symptom
- `TypeError` similar to: "Outcome names should be of type str, got: <class 'int'>"

### Root cause
- `skrub.choose_from({...})` was used with non-string dictionary keys.

### Correct pattern
```python
# Valid
max_depth = skrub.choose_from({"6": 6, "8": 8, "10": 10}, name="max_depth")

# Also valid for numeric ranges
max_depth = skrub.choose_int(6, 10, name="max_depth")
```

### Avoid
```python
max_depth = skrub.choose_from({6: 6, 8: 8, 10: 10}, name="max_depth")
```

## 5) `keep_subsampling` misuse in `make_learner(...)`

### Symptom
- `ValueError` when `keep_subsampling=True` is used inconsistently.

### Root cause
- Subsampling was not configured, or incompatible `fitted`/subsampling settings are used.

### Correct pattern
- If no prior subsampling setup exists: keep default behavior (`keep_subsampling=False`).
- Use `keep_subsampling=True` only when subsampling is intentionally configured in the plan.

## 6) RMSE incompatibility (`squared=False`)

### Symptom
- `TypeError: got an unexpected keyword argument 'squared'`

### Root cause
- Local scikit-learn version does not accept `squared=False` in `mean_squared_error`.

### Correct pattern
```python
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_true, y_pred) ** 0.5
```

## 7) Ablation summary receives empty payload

### Symptom
- "No ablation study results are available to summarize."

### Root cause
- Upstream ablation stage produced no runnable variant outputs/metrics.

### Correct pattern
- Ensure ablation stage always emits:
  - tested variants
  - per-variant validation metric
  - selected best variant and rationale

## 8) Ablation agent returns only tool calls (`list_skills`) and stops

### Symptom
- Ablation script file is empty or non-runnable.
- Runtime shows tool calls (e.g. `list_skills`) but no final Python ablation script.

### Root cause
- The model performs skill discovery but does not transition back to code generation.
- In this case, the refinement pipeline receives no usable ablation output and downstream stages can fail or degrade.

#### Why ablation stops at list_skills() (evidence-based)
- Your diagnosis was mostly correct, and there is one additional key runtime behavior:

`refinement/agent.py`
Lines 349-356
```python
ablation_agent = agents.Agent(
    ...
    before_model_callback=check_ablation_finish,
    after_model_callback=functools.partial(
        debug_util.get_code_from_response,
        do_eval=not use_data_leakage_checker,
    ),
)
```
- If the model returns tool output (or no code text), get_code_from_response stores empty code and still evaluates it. For ablation agents, empty Python can execute with returncode=0, and then:

`refinement/agent.py`
Lines 237-250
```python
def check_ablation_finish(...):
    ...
    result_dict = callback_context.state.get(
        f"ablation_code_exec_result_{step}_{task_id}", {}
    )
    if result_dict.get("returncode", 1) == 0:
        return llm_response_module.LlmResponse()
```
- So the loop can consider ablation “finished” even with empty/non-useful script output.
- This matches your artifact: workspace/.../ablation_0.py is empty.

TLDR:
What’s likely happening now:

ablation agent has tools, so the model may first emit tool calls (list_skills, maybe load_skill).
if it does not follow up with Python code in the same generation cycle, after_model_callback still processes response text.
empty/non-code text can end up as an empty script.
empty script execution can return success (returncode=0), and check_ablation_finish can treat ablation as finished early.

### Correct pattern
- Tool calls are prep only, never the final answer.
- Always return runnable ablation code after tool usage.
- Minimum required outputs from the ablation script:
  - one metric line per variant
  - one final best-variant summary line

### Quick remediation checklist
- Confirm ablation prompt explicitly says: "tool calls are preparation only; final output must be executable Python."
- Confirm generated `ablation_*.py` is non-empty.
- Confirm ablation run stdout contains variant metrics and a best-variant summary.
- If missing, rerun with tighter ablation prompt contract before changing core loop logic.

## 9) Refinement loop stability fix (implemented)

### Root cause
- Tool-enabled refinement agents can emit tool calls first (`list_skills`, `load_skill`) with no final code in the same response.
- These tool/prose outputs were being treated as runnable code artifacts in some stages.
- Downstream refinement logic then tried to compare improvement scores on missing/invalid execution results, causing `KeyError: 'score'`.

### Specific fix (surgical)
- In `shared_libraries/code_util.py`:
  - added execution gating for `ablation*` and `plan_implement*` so code is only executed when output is:
    - non-empty, and
    - syntactically valid Python (`compile(..., "exec")` succeeds).
- In `sub_agents/refinement/agent.py`:
  - `check_plan_implement_finish(...)` now requires a successful scored result (`returncode == 0` and `"score"` present) before finishing.
  - plan refinement/outer-loop scoring skips invalid entries missing `score` instead of indexing blindly.
  - empty plan-refine responses are ignored (not appended as candidate plans).

### TL;DR
- Tool calls are now treated as preparation, not executable artifacts.
- Refinement stages wait for real Python output + scored execution before progressing.
- This removes the false-progress path and prevents `KeyError: 'score'` without refactoring core loop architecture.

## 10) Empty tool calls
1) Is your current mental model correct?
Mostly yes:

- get_text_from_response() effectively ignores non-text/tool parts.
- For tool-enabled code agents, you added execution gates in code_util.get_run_code_condition(): non-empty text syntactically valid Python
- You also tightened finish checks in relevant agent.py callbacks to only accept successful scored runs (returncode == 0 + score).
- That combination prevents tool-only turns from being treated as completed code outputs.

Nuance you should keep in mind
- In debug_util.get_code_from_response(), empty text can still be written into state as code candidate before evaluation.
- Your new gates prevent execution/false completion, so correctness is protected.
- But state can still temporarily hold empty code text (usually harmless now, but good to know).
So you’re not missing much — that’s the key architecture.