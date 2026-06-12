# Lessons learned (self-improvement log)

Rules derived from corrections and root-cause findings. Review at session start.

## 2026-06-12 — P1 tuning search failure

1. **Repro the documented pattern before blaming agent compliance.**
   The r7 tuning failure looked like the agent ignoring skill doc #16; a 20-line
   repro in the project venv showed the doc itself prescribed an impossible
   pattern (inline `choose_*` cannot work for CatBoost). When an agent
   repeatedly fails to follow a documented pattern, first verify the pattern
   actually works in the runtime environment — repro scripts are cheap.

2. **skrub fact (verified, skrub 0.9.0):** `choose_*` inside estimator
   constructor kwargs is only resolved for `sklearn.base.BaseEstimator`
   subclasses (sklearn, LightGBM, XGBoost). CatBoost is not one; choices leak
   into `fit` and crash (`NumericChoice is not JSON serializable`). Working
   alternative: `choose_from({label: Estimator(**params)})` over pre-built
   variants; winner label via `search.results_.iloc[0][<choice_name>]`.

3. **Persistent agent drift usually signals an impossible instruction, not a
   weak prompt.** Debug swapped CatBoost → RandomForest/Ridge because dropping
   the backbone was the only change that could make the broken pattern run.
   Fix the mechanism the instructions demand before hardening the prompt.

4. **Attribute drift before fixing it.** The 2026-06-12 run showed ablation
   "drift" was actually the debug agent regenerating a good script from
   scratch (CatBoost variant study → single HGB fit) to fix a trivial missing
   `read_csv`. Trace the log to the exact agent turn that introduced the
   regression before choosing where the fix goes — here the fix belonged in
   the debug contract (smallest-possible-change + backbone injection), not in
   the ablation prompt.

5. **A correct script can still fail the stage on compute budget.** The
   Pattern 4 tune search was textbook-correct but needed > 600s (full-capacity
   boosted-tree fits × n_iter). Search-stage scripts must be budgeted
   (reduced iterations, n_jobs=1 for internally-threaded estimators), with
   capacity restored at bake.

6. **`str.replace("", x)` is a context bomb.** An empty needle inserts the
   replacement between every character of the haystack. The 1.69M-token crash
   was exactly this: `init_plan` replied with prose, `refine_code_block` was
   stored as `""`, and the block merge produced an 8 MB "script" that flowed
   into the debug prompt. Any `a.replace(b, c)` where `b` comes from model
   output needs an emptiness guard — and bound subprocess stdout/stderr
   *after* extracting what downstream consumers need, never before.

7. **Positional remapping needs a key-identity guard.**
   `map_tuning_best_params` sorted raw keys and assigned by plan order; when a
   script already printed human-named params, values got scrambled
   (alphabetical vs plan order). When remapping dicts by order, return early
   if the key sets already match.
