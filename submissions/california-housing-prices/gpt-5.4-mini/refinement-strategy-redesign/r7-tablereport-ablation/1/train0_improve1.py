
import os
import warnings
import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_dataops_pipeline(train_df, target_col="median_house_value"):
    data = skrub.var("data", train_df)

    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    # Proper DataOps-friendly preprocessing: vectorize the full feature table.
    # This keeps the pipeline structure while avoiding the invalid "passthrough" string.
    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        random_state=0,
    )

    pred = X_vec.skb.apply(model, y=y)
    return pred


def main():
    train_df, test_df = load_data()

    target_col = "median_house_value"
    train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)

    pred = build_dataops_pipeline(train_part, target_col=target_col)
    learner = pred.skb.make_learner(fitted=True)

    val_pred = learner.predict({"data": val_part})
    y_val = val_part[target_col].to_numpy()
    final_validation_score = mean_squared_error(y_val, val_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    full_pred = build_dataops_pipeline(train_df, target_col=target_col)
    full_learner = full_pred.skb.make_learner(fitted=True)
    test_predictions = full_learner.predict({"data": test_df})

    submission = pd.DataFrame({"median_house_value": np.asarray(test_predictions).ravel()})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
