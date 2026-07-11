
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

random_state = 42
test_size = 0.2
target_col = "yield"
n_iter = 5
n_jobs = 1

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_variants = {
    "d4_lr0.03": dict(depth=4, learning_rate=0.03, iterations=500, verbose=0),
    "d6_lr0.03": dict(depth=6, learning_rate=0.03, iterations=500, verbose=0),
    "d6_lr0.05": dict(depth=6, learning_rate=0.05, iterations=500, verbose=0),
    "d8_lr0.03": dict(depth=8, learning_rate=0.03, iterations=500, verbose=0),
}
model = skrub.choose_from(
    {k: CatBoostRegressor(loss_function="MAE", random_seed=random_state, **p) for k, p in cat_variants.items()},
    name="model_variant",
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=n_iter, n_jobs=n_jobs, random_state=random_state, fitted=True
)
search.fit({"data": train_part})

valid_pred = np.asarray(search.best_learner_.predict({"data": valid_part}), dtype=float).ravel()
final_validation_score = mean_absolute_error(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

chosen_variant = search.results_.iloc[0]["model_variant"]
best_params = dict(cat_variants[chosen_variant])
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
