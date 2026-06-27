What gets written in 1/ and when
Folder workspace/<task>/1/ = task_id "1" (first refinement task). Scripts are written when an agent runs and executes code (code_util.evaluate_code → run_python_code).

File	Stage	Agent / event
init_code_0.py, init_code_1.py, …
Initialization
model_eval_* — each init candidate
train0_0.py, train0_1.py, …
Initialization
merger_* — merger references per candidate
train0.py
Initialization
select_best_solution — winning init script
ablation_0.py
Refinement (step 0)
ablation_agent_*
train0_improve0.py, train0_improve1.py, …
Refinement (step 0)
plan_implement_* — inner-loop structural edits
train1.py
Refinement (end of step 0)
update_outer_loop_states — promote or copy train0
train_tune_search.py
Tuning
tune_implement_*
train_tune_baked.py
Tuning
tune_bake_*
train1_tuned.py
Tuning (only if tune beats structural)
promote_tuning_winner — not present on run2
If you had a second refine outer loop (refine_step = 1), you’d also see ablation_1.py, train1_improve*.py, train2.py, etc. Phase-1 run2 only needed one outer step.