
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

train_raw = pd.read_csv(train_path)
test_raw = pd.read_csv(test_path)

def preprocess_solution_1(df):
    df = df.copy()

    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    name_split = df["Name"].astype(str).str.split(" ", n=1, expand=True)
    df["FirstName"] = name_split[0]
    df["LastName"] = name_split[1]

    family_id = df["LastName"].fillna("Unknown").astype(str)
    fam_counts = family_id.map(family_id.value_counts())
    df["FamilySize"] = fam_counts

    age_bins = [-1, 12, 18, 30, 45, 60, 120]
    df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=False)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["SpenderCount"] = (df[spend_cols] > 0).sum(axis=1)

    df["DeckGroup"] = df["Deck"].replace({"A": "Upper", "B": "Upper", "C": "Upper", "T": "Upper",
                                          "D": "Middle", "E": "Middle", "F": "Lower", "G": "Lower"})

    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["NameKnown"] = df["Name"].notna().astype(int)

    df.drop(columns=["Cabin", "Name", "FirstName", "LastName"], inplace=True, errors="ignore")
    return df

def preprocess_solution_2(df):
    df = df.copy()

    cabin_split = df["Cabin"].astype(str).str.split("/", expand=True)
    df["Deck2"] = cabin_split[0]
    df["CabinNum2"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side2"] = cabin_split[2]

    name_split = df["Name"].astype(str).str.split(" ", n=1, expand=True)
    df["Title"] = name_split[0]
    df["Surname"] = name_split[1]

    df["SurnameKnown"] = df["Surname"].notna().astype(int)
    df["TitleKnown"] = df["Title"].notna().astype(int)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["SpendMean"] = df[spend_cols].mean(axis=1)
    df["SpendStd"] = df[spend_cols].std(axis=1)
    df["HasAnySpend"] = (df[spend_cols].fillna(0).sum(axis=1) > 0).astype(int)
    df["ZeroSpendCount"] = (df[spend_cols].fillna(0) == 0).sum(axis=1)

    df["AgeBin2"] = pd.cut(df["Age"], bins=[-1, 5, 12, 18, 25, 35, 50, 70, 120], labels=False)
    df["DeckKnown2"] = df["Cabin"].notna().astype(int)

    df.drop(columns=["Cabin", "Name", "Title", "Surname"], inplace=True, errors="ignore")
    return df

def prepare_xy(df, preprocess_fn):
    dfp = preprocess_fn(df)
    if "Transported" in dfp.columns:
        y = dfp["Transported"].astype(int)
        X = dfp.drop(columns=["Transported"])
    else:
        y = None
        X = dfp

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Missing")
    for col in X.columns:
        if col not in cat_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y, cat_cols

def fit_predict_catboost(X_train, y_train, X_valid, X_test, cat_cols, random_seed=42):
    model = CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_seed,
        verbose=0,
        allow_writing_files=False
    )
    model.fit(
        X_train,
        y_train,
        cat_features=[X_train.columns.get_loc(c) for c in cat_cols],
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )
    val_prob = model.predict_proba(X_valid)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    return model, val_prob, test_prob

def fit_full_catboost(X, y, X_test, cat_cols, random_seed=42):
    model = CatBoostClassifier(
        iterations=2000,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=random_seed,
        verbose=0,
        allow_writing_files=False
    )
    model.fit(
        X,
        y,
        cat_features=[X.columns.get_loc(c) for c in cat_cols]
    )
    test_prob = model.predict_proba(X_test)[:, 1]
    return model, test_prob

# Shared holdout split for final validation metric
train_idx, valid_idx = train_test_split(
    np.arange(len(train_raw)),
    test_size=0.2,
    random_state=42,
    stratify=train_raw["Transported"]
)

train_part = train_raw.iloc[train_idx].reset_index(drop=True)
valid_part = train_raw.iloc[valid_idx].reset_index(drop=True)

# Base learner 1
X_train1, y_train1, cat_cols1 = prepare_xy(train_part, preprocess_solution_1)
X_valid1, y_valid1, _ = prepare_xy(valid_part, preprocess_solution_1)
X_test1, _, _ = prepare_xy(test_raw, preprocess_solution_1)

# Base learner 2
X_train2, y_train2, cat_cols2 = prepare_xy(train_part, preprocess_solution_2)
X_valid2, y_valid2, _ = prepare_xy(valid_part, preprocess_solution_2)
X_test2, _, _ = prepare_xy(test_raw, preprocess_solution_2)

# Train each base learner on the same holdout split
model1 = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
    allow_writing_files=False
)
model1.fit(
    X_train1,
    y_train1,
    cat_features=[X_train1.columns.get_loc(c) for c in cat_cols1],
    eval_set=(X_valid1, y_valid1),
    use_best_model=True
)
valid_prob1 = model1.predict_proba(X_valid1)[:, 1]
test_prob1 = model1.predict_proba(X_test1)[:, 1]

model2 = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=0,
    allow_writing_files=False
)
model2.fit(
    X_train2,
    y_train2,
    cat_features=[X_train2.columns.get_loc(c) for c in cat_cols2],
    eval_set=(X_valid2, y_valid2),
    use_best_model=True
)
valid_prob2 = model2.predict_proba(X_valid2)[:, 1]
test_prob2 = model2.predict_proba(X_test2)[:, 1]

# Simple soft voting
avg_valid_prob = (valid_prob1 + valid_prob2) / 2.0
avg_test_prob = (test_prob1 + test_prob2) / 2.0

# Optional tiny step beyond averaging: logistic blender on validation probabilities
meta_X_valid = np.column_stack([valid_prob1, valid_prob2])
meta_X_test = np.column_stack([test_prob1, test_prob2])

blender = LogisticRegression(max_iter=1000)
blender.fit(meta_X_valid, y_valid1.astype(int))
blend_valid_prob = blender.predict_proba(meta_X_valid)[:, 1]
blend_test_prob = blender.predict_proba(meta_X_test)[:, 1]

# Choose best ensemble rule on hold-out validation
avg_valid_pred = (avg_valid_prob >= 0.5).astype(int)
blend_valid_pred = (blend_valid_prob >= 0.5).astype(int)

avg_acc = accuracy_score(y_valid1.astype(int), avg_valid_pred)
blend_acc = accuracy_score(y_valid1.astype(int), blend_valid_pred)

if blend_acc >= avg_acc:
    final_valid_prob = blend_valid_prob
    final_test_prob = blend_test_prob
    final_validation_score = blend_acc
else:
    final_valid_prob = avg_valid_prob
    final_test_prob = avg_test_prob
    final_validation_score = avg_acc

final_valid_pred = (final_valid_prob >= 0.5).astype(int)
final_validation_score = accuracy_score(y_valid1.astype(int), final_valid_pred)

print(f"Final Validation Performance: {final_validation_score}")

# Train full-data models for submission
full_X1, full_y1, full_cat_cols1 = prepare_xy(train_raw, preprocess_solution_1)
full_X2, full_y2, full_cat_cols2 = prepare_xy(train_raw, preprocess_solution_2)
full_test_X1, _, _ = prepare_xy(test_raw, preprocess_solution_1)
full_test_X2, _, _ = prepare_xy(test_raw, preprocess_solution_2)

full_model1, full_test_prob1 = fit_full_catboost(full_X1, full_y1, full_test_X1, full_cat_cols1, random_seed=42)
full_model2, full_test_prob2 = fit_full_catboost(full_X2, full_y2, full_test_X2, full_cat_cols2, random_seed=42)

full_avg_test_prob = (full_test_prob1 + full_test_prob2) / 2.0
full_meta_X_test = np.column_stack([full_test_prob1, full_test_prob2])
full_blender = LogisticRegression(max_iter=1000)
full_meta_X_train = np.column_stack([
    model1.predict_proba(X_valid1)[:, 1],
    model2.predict_proba(X_valid2)[:, 1]
])
full_blender.fit(full_meta_X_train, y_valid1.astype(int))
full_blend_test_prob = full_blender.predict_proba(full_meta_X_test)[:, 1]

# Use the same rule selected on validation, but on full-data test predictions
if blend_acc >= avg_acc:
    test_pred = (full_blend_test_prob >= 0.5).astype(bool)
else:
    test_pred = (full_avg_test_prob >= 0.5).astype(bool)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv(test_path)["PassengerId"],
    "Transported": test_pred
})
submission.to_csv("submission.csv", index=False)
