
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "revenue"
    random_state = 42

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=random_state
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_state=random_state,
        verbose=0,
    )

    cat_features = ["Open Date", "City", "City Group", "Type"]

    X_train = X_train.skb.apply(vectorizer)
    predictor = X_train.skb.apply(model, y=y_train)

    val_learner = predictor.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})
    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
