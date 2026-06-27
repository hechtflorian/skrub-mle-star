# Phase-1 speed results — honest explanation

Why did **skrub-full** show shorter **wall time** and **Python execution time** than vanilla MLE-STAR in phase-1? This note interprets the archived runs without over-claiming causality.

**Sources:** [`experiments/phase1/results/phase1_summary.md`](../experiments/phase1/results/phase1_summary.md), [`phase1_all_runs_12.csv`](../experiments/phase1/results/phase1_all_runs_12.csv).

---

## Short answer

**We cannot fairly claim “DataOps pipelines are faster than sklearn.”** Phase-1 shows skrub-full runs were **faster on average**, but that mostly reflects **different generated Python** (lighter training scripts in refine/ensemble), not a proven speed property of skrub graphs themselves. The comparison is **confounded**: different prompts, different code paths, small sample size, and skrub-full has an **extra tuning stage** vanilla lacks.

---

## What we measured

| Metric | Source | Meaning |
|--------|--------|---------|
| **Python exec** | Sum of `execution_time` on scored `*_exec_result_*` in `final_state.json` | Wall time of each `subprocess.run(python …)`; **summed** per stage (includes retries, ablation, tune attempts) |
| **Wall time** | ADK log `Script started on` → `Script done on` | Full session: LLM calls, tools/skills, waiting, **plus** all Python exec |
| **Python fraction** | `exec_seconds / wall_seconds` | Share of the run spent in Python vs agent overhead |

**Key finding:** On housing (clean runs), **scored Python runs per stage were the same** for vanilla and skrub-full — skrub did **not** win by running fewer scripts. It won because **each script tended to finish faster**, especially in **refinement and ensemble**.

---

## Where the gap comes from (California housing, clean runs)

Housing aggregates exclude probable leakage runs: **skrub run1**, **vanilla run2** (`n=2` per system).

| Stage | Vanilla exec (mean) | Skrub-full exec (mean) |
|-------|---------------------|-------------------------|
| Initialization | ~108 s | ~113 s (**~tie**) |
| **Refinement** | **~443 s** | **~68 s** |
| Tuning | — | ~143 s (skrub only) |
| **Ensemble** | **~441 s** | **~88 s** |
| **Total exec** | **~1 221 s** | **~462 s** (~**2.6×** less) |
| **Wall time** | **~1 844 s** | **~786 s** (~**2.3×** shorter) |

Init cost is essentially the same. Most savings are in **refine + ensemble Python**, not initialization.

Example per-run totals from `phase1_all_runs_12.csv`:

| Run | System | Python exec | Wall |
|-----|--------|-------------|------|
| vanilla run3 | vanilla | **1 629 s** | 1 864 s |
| vanilla run1 | vanilla | 813 s | 1 823 s |
| skrub run2 | skrub-full | **274 s** | 693 s |
| skrub run3 | skrub-full | 651 s | 879 s |

High variance within vanilla (run1 vs run3 exec differs ~2×) — treat aggregates cautiously.

### Spaceship Titanic (`n=3` each)

| | Vanilla | Skrub-full |
|--|---------|------------|
| Total Python exec | ~2 243 s | ~1 298 s (~**42%** less) |
| Wall time | ~2 905 s | ~2 388 s (~**18%** shorter) |
| Python fraction | ~**76%** | ~**55%** |

Effect is **real but smaller** on Titanic; vanilla sessions spend a **larger share** of wall time in Python (longer model training).

---

## Is DataOps / skrub inherently faster than plain sklearn?

**Probably not in general — and we did not prove that it is.**

Skrub-full still runs CatBoost, `TableVectorizer`, `train_test_split`, etc. The DataOps layer adds graph building and learner wiring; that is **structure**, not a free speedup.

We did **not** run a controlled micro-benchmark like:

> same model, same iterations, same data — only “skrub graph” vs “identical sklearn script”

What we compared is **two full agent systems** that produce **different code**.

---

## Plausible explanations (most → least confident)

### 1. Lighter generated training code (main story)

Vanilla refine/ensemble scripts on these runs often do **heavier work**: long CatBoost fits, grids, multi-model blends, extra refits (e.g. vanilla housing run3 with ~1.6k s Python exec).

Skrub-full scripts tend toward a **compact template**:

- `TableVectorizer` + **one** CatBoost
- Often `iterations=500`, `verbose=0`
- Tuning search with **bounded** configs (e.g. short `choose_from` grid in archived tune scripts)
- Feature engineering in one `@skrub.deferred` block

That yields **shorter wall-clock per script**, regardless of whether `.skb.apply` is faster than `.fit`.

### 2. Skrub constraints may reduce accidental heaviness (indirect)

Skills and prompts nudge agents toward:

- bounded tune search (P5 budget rules)
- avoiding unbounded `GridSearchCV` in the DataOps graph
- `verbose=0` on boosting models

So skrub-full may produce **less expensive Python by convention**, not because the paradigm is intrinsically quicker.

### 3. Ensemble stage difference

On housing, vanilla ensemble **~441 s** vs skrub **~88 s** (mean) is a large slice of total exec. Vanilla ensemble scripts likely did **more fitting/blending** than skrub ensemble scripts on those runs. We did not micro-profile each `ensemble/final_solution.py`; stage totals strongly suggest this.

### 4. Wall time follows Python; LLM is not the whole story

Skrub had **more refinement debug** on average, yet **shorter wall** on housing — vanilla spent a larger share of the session in slow Python (~66% vs ~57% Python fraction).

**Faster Python drove shorter wall**, not fewer LLM turns.

---

## What we should not over-claim

| Claim | Verdict |
|-------|---------|
| “DataOps is faster than sklearn” | **Not supported** |
| “Skrub always runs fewer scripts” | **False** on housing (same exec counts per stage) |
| “Skrub is faster because fewer debugs” | **False** — often **more** debug on skrub |
| “Effect is uniform across stages” | **False** — init ~tie; Titanic gap smaller |
| “Controlled experiment” | **No** — different prompts, extra tuning stage, different generated code |

---

## Safe wording for presentations

> Phase-1 skrub-full sessions used **less total Python time**, mainly because **refinement and ensemble scripts trained lighter models** (shorter CatBoost runs, simpler pipelines), not because we ran fewer stages. Init cost was similar. We **cannot** attribute the win to DataOps itself — we compared two agent stacks that **generate different code**. The operational benefit may be **more predictable, bounded pipelines** via skills and structure, which **correlated with** faster runs on these two tasks — but with **n=2–3** and high run-to-run variance.

If asked why refine/ensemble specifically:

> Ablation + implement scripts under skrub often use a **fixed short template** (e.g. 500-iteration CatBoost, single vectorizer). Vanilla refine/ensemble on the same runs sometimes spent **much longer in Python**. That is the largest measured gap in phase-1.

---

## What would be needed for stronger causal claims

1. **Controlled micro-benchmark:** same task, same backbone + iterations, skrub graph vs hand-written sklearn — timed outside the agent loop.
2. **Script-level profiling:** fit vs predict vs import from archived workspaces (`experiments/phase1/.../1/`, `ensemble/`).
3. **More tasks / repeats** to shrink variance (vanilla housing run3 vs run1 exec differs substantially).

---

## Bottom line

The phase-1 speed result is **real in the logs** but **misleading if summarized as “DataOps is faster.”** The honest story: skrub-full produced **faster-running Python in refine/ensemble** on California housing (and somewhat on Titanic), while init looked similar; wall time dropped mainly because vanilla sessions were **Python-heavy** and those scripts were **heavier**. Whether that repeats depends on agent behavior and prompts, not on skrub being intrinsically quicker than sklearn. Exact per-script causes were **not** profiled in phase-1 — stage-level aggregates are the best evidence we have.
