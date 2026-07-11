
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")

target_col = "Rings"
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

model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
y_true = valid_part[target_col].to_numpy()
y_pred = np.asarray(valid_pred)

final_validation_score = float(
    np.sqrt(
        np.mean(
            (
                np.log1p(np.maximum(y_pred, 0))
                - np.log1p(np.maximum(y_true, 0))
            ) ** 2
        )
    )
)

print(f"Final Validation Performance: {final_validation_score}")
