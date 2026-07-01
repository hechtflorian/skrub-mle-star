
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import skrub

warnings.filterwarnings("ignore")


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    sample_path = os.path.join("./input", "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    # Build the DataOps graph first
    data = skrub.var("data", train_df)
    X = data.drop(columns=[target_col]).skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Keep the pipeline structure in skrub DataOps
    X_train_op, X_val_op, y_train_op, y_val_op = X.skb.train_test_split(
        y=y, test_size=0.2, random_state=42
    )

    model = HistGradientBoostingRegressor(random_state=42)
    pred_op = X_train_op.skb.apply(model, y=y_train_op)

    fitted_model = pred_op.eval()

    # Fix: X_val_op / y_val_op may be plain dict/objects, so evaluate safely
    val_X = X_val_op.eval() if hasattr(X_val_op, "eval") else X_val_op
    val_y = y_val_op.eval() if hasattr(y_val_op, "eval") else y_val_op

    # If the split returns wrapped objects, unwrap them recursively when needed
    if hasattr(val_X, "skb") and hasattr(val_X.skb, "eval"):
        val_X = val_X.skb.eval()
    if hasattr(val_y, "skb") and hasattr(val_y.skb, "eval"):
        val_y = val_y.skb.eval()

    val_pred = fitted_model.predict(val_X)
    final_validation_score = mean_squared_error(val_y, val_pred, squared=False)
    print(f"Final Validation Performance: {final_validation_score}")

    # Final model on all training data for submission
    final_model = HistGradientBoostingRegressor(random_state=42)
    final_model.fit(train_df.drop(columns=[target_col]), train_df[target_col])

    test_pred = final_model.predict(test_df)

    if os.path.exists(sample_path):
        sub = pd.read_csv(sample_path)
        sub[target_col] = test_pred
    else:
        sub = pd.DataFrame({target_col: test_pred})

    sub.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
