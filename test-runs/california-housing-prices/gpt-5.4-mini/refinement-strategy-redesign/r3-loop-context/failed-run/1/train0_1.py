
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(train_df, test_df):
    target_col = "median_house_value"

    train_split, valid_split = train_test_split(
        train_df, test_size=0.2, random_state=42
    )

    # -------------------------
    # Validation data and target
    # -------------------------
    valid_X = valid_split.drop(columns=[target_col], errors="ignore")
    y_valid = valid_split[target_col].values

    # -------------------------
    # Base model: CatBoost
    # -------------------------
    cat_data = skrub.var("data", train_split)
    cat_X = cat_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    cat_y = cat_data[target_col].skb.mark_as_y()

    cat_vectorizer = skrub.TableVectorizer()
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1000,
        depth=8,
        learning_rate=0.03,
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
    )

    cat_plan = cat_X.skb.apply(cat_vectorizer).skb.apply(cat_model, y=cat_y)
    cat_learner = cat_plan.skb.make_learner(fitted=True)
    cat_valid_preds = cat_learner.predict({"data": valid_X})

    # -------------------------
    # Reference model: HGBR
    # -------------------------
    hgb_data = skrub.var("data", train_split)
    hgb_X = hgb_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    hgb_y = hgb_data[target_col].skb.mark_as_y()

    hgb_vectorizer = skrub.TableVectorizer()
    hgb_model = HistGradientBoostingRegressor(random_state=42)

    hgb_plan = hgb_X.skb.apply(hgb_vectorizer).skb.apply(hgb_model, y=hgb_y)
    hgb_learner = hgb_plan.skb.make_learner(fitted=True)
    hgb_valid_preds = hgb_learner.predict({"data": valid_X})

    # -------------------------
    # Simple ensemble on holdout
    # -------------------------
    valid_preds = 0.6 * cat_valid_preds + 0.4 * hgb_valid_preds
    final_validation_score = mean_squared_error(y_valid, valid_preds) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # -------------------------
    # Fit on full data for test inference
    # -------------------------
    full_data_cat = skrub.var("data", train_df)
    full_X_cat = full_data_cat.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    full_y_cat = full_data_cat[target_col].skb.mark_as_y()
    full_cat_plan = full_X_cat.skb.apply(cat_vectorizer).skb.apply(cat_model, y=full_y_cat)
    full_cat_learner = full_cat_plan.skb.make_learner(fitted=True)

    full_data_hgb = skrub.var("data", train_df)
    full_X_hgb = full_data_hgb.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    full_y_hgb = full_data_hgb[target_col].skb.mark_as_y()
    full_hgb_plan = full_X_hgb.skb.apply(hgb_vectorizer).skb.apply(hgb_model, y=full_y_hgb)
    full_hgb_learner = full_hgb_plan.skb.make_learner(fitted=True)

    cat_test_preds = full_cat_learner.predict({"data": test_df})
    hgb_test_preds = full_hgb_learner.predict({"data": test_df})

    test_preds = 0.6 * cat_test_preds + 0.4 * hgb_test_preds
    return test_preds


def main():
    train_df, test_df = load_data()
    preds = build_pipeline(train_df, test_df)
    submission = pd.DataFrame({"median_house_value": preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
