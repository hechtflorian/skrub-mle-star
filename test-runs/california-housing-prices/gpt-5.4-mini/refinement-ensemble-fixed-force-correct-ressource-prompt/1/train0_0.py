
import os
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "median_house_value"

data = skrub.var("data", train)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    high_cardinality=skrub.choose_from(
        {
            "minhash": skrub.MinHashEncoder(n_components=16),
            "gap": skrub.GapEncoder(n_components=16),
        },
        name="high_card_encoder",
    )
)

model = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.03, 0.12, log=True, name="learning_rate"),
    max_depth=skrub.choose_int(4, 8, name="max_depth"),
    min_samples_leaf=skrub.choose_int(15, 50, name="min_samples_leaf"),
    random_state=42,
)

pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
learner = pred.skb.make_learner(fitted=True)

n = len(train)
val_size = max(1, int(0.2 * n))
train_idx = train.index[:-val_size]
val_idx = train.index[-val_size:]

train_fit = train.loc[train_idx].reset_index(drop=True)
val_fit = train.loc[val_idx].reset_index(drop=True)

fit_learner = pred.skb.make_learner(fitted=True)
fit_learner = pred.skb.make_learner(fitted=False)
fit_learner = pred.skb.make_learner(fitted=True)

train_only = pd.concat([train_fit], axis=0).reset_index(drop=True)
learner = pred.skb.make_learner(fitted=True)
learner = pred.skb.make_learner(fitted=False)

train_data = train.iloc[:-val_size].reset_index(drop=True)
val_data = train.iloc[-val_size:].reset_index(drop=True)

train_graph = skrub.var("data", train_data)
X_train = train_graph.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = train_graph[target_col].skb.mark_as_y()

train_pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
trained_learner = train_pred_graph.skb.make_learner(fitted=True)

val_pred = trained_learner.predict({"data": val_data})
val_rmse = mean_squared_error(val_data[target_col], val_pred) ** 0.5
print(f"Final Validation Performance: {val_rmse}")

test_pred = trained_learner.predict({"data": test})
submission = pd.DataFrame({target_col: test_pred})
submission.to_csv("submission.csv", index=False)
