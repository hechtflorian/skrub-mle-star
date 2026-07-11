
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
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

model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)

valid_X = valid_part.drop(columns=target_col, errors="ignore")
valid_pred = learner.predict({"data": valid_X})

y_true = valid_part[target_col].to_numpy()
y_pred = np.asarray(valid_pred)
final_validation_score = float(np.sqrt(np.mean((np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_true, 0))) ** 2)))

print(f"Final Validation Performance: {final_validation_score}")
