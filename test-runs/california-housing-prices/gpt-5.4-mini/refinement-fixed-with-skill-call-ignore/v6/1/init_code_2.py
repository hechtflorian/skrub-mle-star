
import os
import warnings

import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TARGET = "median_house_value"
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

data = skrub.var("data", train_part)
X = data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = RandomForestRegressor(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=2,
    random_state=0,
    n_jobs=-1,
)

pred = X_vec.skb.apply(model, y=y)
learner = pred.skb.make_learner(fitted=True)

valid_features = valid_part.drop(columns=[TARGET], errors="ignore")
valid_preds = learner.predict({"data": valid_features})
final_validation_score = mean_squared_error(valid_part[TARGET], valid_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()

full_pred = full_X.skb.apply(vectorizer).skb.apply(model, y=full_y)
full_learner = full_pred.skb.make_learner(fitted=True)

test_preds = full_learner.predict({"data": test_df})
submission = pd.DataFrame({TARGET: test_preds})
submission.to_csv("submission.csv", index=False)
