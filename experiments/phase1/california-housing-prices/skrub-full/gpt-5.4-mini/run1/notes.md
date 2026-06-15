- seemed to not be able to correctly install models (catboost) and possible fell back to histgradientboost silently. score still good
- ablation struggled, and debug might have rendered ablation code kinda useless
- leakage!

Is the 42k score valid? Mostly no — optimistic validation, not classic FE leakage
What's clean:

Model A uses a proper 80/20 split (random_state=42).
TableVectorizer is fit on train_part only via the skrub pipeline; validation rows are predicted through learner.predict({"data": valid_part}). No target-in-features bug.
What's wrong (three issues in final_solution.py):

1. Cross-split leakage in ensemble member B (biggest issue)


final_solution.py
Lines 58-75
train_idx_b, valid_idx_b = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=7
)
...
valid_pred_b = val_learner_b.predict({"data": valid_part})
Model B trains on the 80% from split seed 7, but is evaluated on valid_part from split seed 42. Those splits differ, so ~82.5% of validation rows (396/480) were in B's training set. B's RMSE on that set collapses to ~33.7k vs ~55.8k on its own honest holdout.

2. Validation-set calibration


final_solution.py
Lines 94-127
sorted_target = np.sort(valid_part[target_col].to_numpy())
...
valid_blend_cal, blend_coef = linear_calibrate(valid_blend_pred, valid_part[target_col].to_numpy())
...
use_median = rmse_median < rmse_blend
It maps ranks to validation target quantiles, fits linear calibration on validation labels, and chooses blend method on the same holdout. That further lowers reported RMSE without being a fair generalization estimate.