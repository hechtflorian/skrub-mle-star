
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def predictor_lgbm(data_train, target_col):
    X_train = data_train.drop(columns=["id"], errors="ignore").drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    predictor = X_train.skb.apply(vectorizer).skb.apply(
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        ),
        y=y_train,
    )
    return predictor


def main():
    train_path = os.path.join(".", "input", "train.csv")
    train_df = pd.read_csv(train_path)

    target_col = "NObeyesdad"
    random_state = 42

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=random_state,
        stratify=train_df[target_col],
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    data_train = skrub.var("data", train_part)
    predictor = predictor_lgbm(data_train, target_col)

    val_learner = predictor.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})

    final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
