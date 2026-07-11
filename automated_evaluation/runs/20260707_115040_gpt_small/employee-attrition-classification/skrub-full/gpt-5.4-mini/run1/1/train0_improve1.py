
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random_state = 42
target_col = "Attrition"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data = skrub.var("data", train_part)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()


import numpy as np
import pandas as pd
import skrub
import skrub.selectors as s
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_structural_features(df):
    out = df.copy()

    # Drop constant columns explicitly via a helper feature-cleaning block
    nunique = out.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        out = out.drop(columns=const_cols)

    # Lightweight ratio / interaction features for numeric health/tenure-style fields
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    # Heuristic, robust pair selection from common tabular health/tenure patterns
    candidate_pairs = [
        ("tenure", "age"),
        ("tenure", "monthlycharges"),
        ("tenure", "totallcharges"),
        ("totalcharges", "tenure"),
        ("monthlycharges", "tenure"),
        ("age", "tenure"),
        ("bmi", "age"),
        ("cholesterol", "age"),
        ("income", "age"),
    ]

    created_any = False
    for a, b in candidate_pairs:
        if a in out.columns and b in out.columns:
            a_vals = pd.to_numeric(out[a], errors="coerce")
            b_vals = pd.to_numeric(out[b], errors="coerce").replace(0, np.nan)

            out[f"{a}_x_{b}"] = (a_vals * pd.to_numeric(out[b], errors="coerce")).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            out[f"{a}_per_{b}"] = (a_vals / b_vals).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            out[f"{b}_per_{a}"] = (pd.to_numeric(out[b], errors="coerce") / a_vals.replace(0, np.nan)).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            created_any = True

    # If explicit name matches are absent, build a compact generic block from numeric columns
    if not created_any and len(num_cols) >= 2:
        base_a = num_cols[0]
        base_b = num_cols[1]
        a_vals = pd.to_numeric(out[base_a], errors="coerce")
        b_vals = pd.to_numeric(out[base_b], errors="coerce").replace(0, np.nan)
        out[f"{base_a}_x_{base_b}"] = (a_vals * pd.to_numeric(out[base_b], errors="coerce")).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        out[f"{base_a}_per_{base_b}"] = (a_vals / b_vals).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    return out


data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(add_structural_features)

X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

# Deliberate routing for categoricals:
# - low-cardinality categoricals -> one-hot
# - higher-cardinality strings -> compact string encoder
low_card_selector = (s.categorical() | s.string()) & s.cardinality_below(20)
high_card_selector = s.string() & ~s.cardinality_below(20)

low_card_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
high_card_encoder = skrub.StringEncoder()

vectorizer = skrub.TableVectorizer(
    low_cardinality=low_card_encoder,
    high_cardinality=high_card_encoder,
)

cat_model = CatBoostClassifier(
    verbose=0,
    random_state=random_state,
    loss_function="Logloss",
    iterations=450,
    depth=7,
    learning_rate=0.06,
    l2_leaf_reg=4.0,
)

lgbm_model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph_cat = X.skb.apply(vectorizer).skb.apply(cat_model, y=y)
pred_graph_lgbm = X.skb.apply(vectorizer).skb.apply(lgbm_model, y=y)

learner_cat = pred_graph_cat.skb.make_learner(fitted=True)
learner_lgbm = pred_graph_lgbm.skb.make_learner(fitted=True)

valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part})).ravel()
valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part})).ravel()

# CatBoost is primary; LightGBM serves as a small fallback comparison, not a wide blend sweep
valid_pred = 0.8 * valid_pred_cat + 0.2 * valid_pred_lgbm
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

