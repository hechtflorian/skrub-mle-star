
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fe_func(df):
    out = df.copy()
    out["Cabin"] = out["Cabin"].fillna("X/0/X").astype(str)
    cabin_split = out["Cabin"].str.split("/", expand=True)
    out["Deck"] = cabin_split[0]
    out["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["Side"] = cabin_split[2]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["TotalSpend"] = out[spend_cols].fillna(0).sum(axis=1)
    out["IsAlone"] = (out["TotalSpend"] == 0).astype(int)
    out["HasSpent"] = (out["TotalSpend"] > 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"].fillna(-1),
        bins=[-2, 0, 12, 18, 30, 45, 60, 200],
        labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    ).astype(str)

    out["FamilySizeFlag"] = (
        out["PassengerId"].astype(str).str.split("_").str[1].astype(int) > 0
    ).astype(int)
    out["LogTotalSpend"] = np.log1p(out["TotalSpend"])
    return out

metric_fn = accuracy_score

def fit_pipeline(train_df_local, seed=42):
    data_train = skrub.var("data", train_df_local)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer(
        low_cardinality=skrub.ToCategorical(),
        high_cardinality=skrub.StringEncoder(),
    )
    model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )

    pred_graph = X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    return learner

def predict_proba_positive(learner, df):
    pred = learner.predict_proba({"data": df})
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred
    if pred.shape[1] == 2:
        return pred[:, 1]
    return pred.reshape(-1)

# Base model on original split
base_learner = fit_pipeline(train_part, seed=42)
base_valid_proba = predict_proba_positive(base_learner, valid_part)

# Cheap deterministic variants via different split seeds
split_seeds = [7, 123, 999]
variant_valid_probas = [base_valid_proba]
variant_weights = []

base_valid_pred = (base_valid_proba > 0.5).astype(int)
base_valid_score = metric_fn(valid_part[target_col].astype(int), base_valid_pred)
variant_weights.append(base_valid_score)

for s in split_seeds:
    tr_idx, va_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=s,
        stratify=train_df[target_col],
    )
    tr_part = train_df.iloc[tr_idx].copy()
    va_part = train_df.iloc[va_idx].copy()
    learner = fit_pipeline(tr_part, seed=s)
    va_proba = predict_proba_positive(learner, valid_part)
    variant_valid_probas.append(va_proba)

    va_pred = (va_proba > 0.5).astype(int)
    va_score = metric_fn(valid_part[target_col].astype(int), va_pred)
    variant_weights.append(va_score)

probas_stack = np.vstack(variant_valid_probas)
p_mean = probas_stack.mean(axis=0)
p_med = np.median(probas_stack, axis=0)

# Conservative agreement-aware blend
agreement = probas_stack.max(axis=0) - probas_stack.min(axis=0)
final_valid_prob = np.where(agreement > 0.25, 0.7 * p_mean + 0.3 * p_med, p_mean)
final_valid_pred = (final_valid_prob > 0.5).astype(int)
final_validation_score = metric_fn(valid_part[target_col].astype(int), final_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: full-train ensemble with same pipeline
full_train_df = train_df.copy()
full_learners = []

# main full-data fit
full_learners.append(fit_pipeline(full_train_df, seed=42))

# two additional bootstrap-style refits on full training set
rng = np.random.default_rng(42)
for seed in [7, 123]:
    boot_idx = rng.choice(len(full_train_df), size=len(full_train_df), replace=True)
    boot_df = full_train_df.iloc[boot_idx].copy()
    full_learners.append(fit_pipeline(boot_df, seed=seed))

test_probas = []
for learner in full_learners:
    test_probas.append(predict_proba_positive(learner, test_df))

test_stack = np.vstack(test_probas)
test_mean = test_stack.mean(axis=0)
test_med = np.median(test_stack, axis=0)
test_agreement = test_stack.max(axis=0) - test_stack.min(axis=0)
final_test_prob = np.where(test_agreement > 0.25, 0.7 * test_mean + 0.3 * test_med, test_mean)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": (final_test_prob > 0.5).astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
