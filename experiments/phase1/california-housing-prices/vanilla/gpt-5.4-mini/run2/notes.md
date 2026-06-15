- leakage in refine (train0_improve0.py)!

Root cause: evaluate validation after training on full data
The promoted refinement script (train0_improve0.py, also copied to train1.py) does this:


train0_improve0.py
Lines 93-110
# Train final model on full data
final_model = CatBoostRegressor(...)
final_model.fit(X, y, verbose=200)
# Final validation performance for parsing
final_val_pred = final_model.predict(X_val)
final_validation_score = root_mean_squared_error(y_val, final_val_pred)
print(f"Final Validation Performance: {final_validation_score}")
Flow:

Grid search on X_tr with early stopping on X_val → honest best ~54,276
Retrain on full X, y (validation rows included)
Predict X_val and print that as Final Validation Performance → ~10,550
The model has already seen validation labels. That’s direct leakage, not subtle tuning bias.

Reproduction:

Grid best (honest): 54276.18
After full-train eval on val: 10550.38 ← matches your archived score exactly