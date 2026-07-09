
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

random_state = 42
test_size = 0.2
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Split for honest holdout validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Smallest possible fix: add imputation before LogisticRegression so NaNs are handled.
vectorizer = skrub.TableVectorizer()
model = SimpleImputer(strategy="most_frequent")  # placeholder to keep line structure? no

# Use an imputer in the DataOps chain before the classifier
pred = X_train.skb.apply(vectorizer).skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(
    LogisticRegression(max_iter=1000),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
