
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

# Keep original DataOps pipeline shape intact
data = skrub.var("data", train_df)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)

predictor = X_vec.skb.apply(model, y=y)

# Hold-out validation split for metric reporting
rng = np.random.RandomState(42)
indices = np.arange(len(train_df))
rng.shuffle(indices)
val_size = max(1, int(0.2 * len(train_df)))
val_idx = indices[:val_size]
tr_idx = indices[val_size:]

train_part = train_df.iloc[tr_idx].copy()
val_part = train_df.iloc[val_idx].copy()

X_tr = train_part.drop(columns=[target_col], errors="ignore")
y_tr = train_part[target_col].copy()
X_val = val_part.drop(columns=[target_col], errors="ignore")
y_val = val_part[target_col].copy()

# Ensemble settings: multi-seed bagging around the same pipeline
seeds = [42, 52, 62, 72, 82]
test_predictions = []
val_predictions = []
val_scores = []

for seed in seeds:
    # Fit separate preprocessing on this run's training subset
    vectorizer_fit = skrub.TableVectorizer()
    X_tr_vec = vectorizer_fit.fit_transform(X_tr)
    X_val_vec = vectorizer_fit.transform(X_val)
    X_test_vec = vectorizer_fit.fit_transform(X_tr).astype(None) if False else None

    # Train the same CatBoost configuration
    eval_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        random_seed=seed,
        verbose=0,
    )
    eval_model.fit(X_tr_vec, y_tr)

    val_pred = eval_model.predict(X_val_vec)
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5

    # Predict on test using this run's fitted vectorizer
    X_test = test_df.copy()
    X_test_vec = vectorizer_fit.transform(X_test)
    test_pred = eval_model.predict(X_test_vec)

    test_predictions.append(test_pred)
    val_predictions.append(val_pred)
    val_scores.append(val_rmse)

# Plain averaging ensemble at prediction time
test_predictions = np.stack(test_predictions, axis=0)
ensemble_test_pred = np.mean(test_predictions, axis=0)

# Report hold-out validation score using the same ensemble averaging
val_predictions = np.stack(val_predictions, axis=0)
ensemble_val_pred = np.mean(val_predictions, axis=0)
final_validation_score = mean_squared_error(y_val, ensemble_val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Train final ensemble on all training data and average predictions
all_test_predictions = []

for seed in seeds:
    vectorizer_final = skrub.TableVectorizer()
    X_all = train_df.drop(columns=[target_col], errors="ignore")
    y_all = train_df[target_col].copy()

    X_all_vec = vectorizer_final.fit_transform(X_all)

    final_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        random_seed=seed,
        verbose=0,
    )
    final_model.fit(X_all_vec, y_all)

    X_test = test_df.copy()
    X_test_vec = vectorizer_final.transform(X_test)
    test_pred = final_model.predict(X_test_vec)
    all_test_predictions.append(test_pred)

all_test_predictions = np.stack(all_test_predictions, axis=0)
final_test_pred = np.mean(all_test_predictions, axis=0)

submission = pd.DataFrame({"median_house_value": final_test_pred})
submission.to_csv("submission.csv", index=False)
