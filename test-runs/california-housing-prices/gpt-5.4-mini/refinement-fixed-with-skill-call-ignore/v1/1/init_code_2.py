
import os
import warnings

import pandas as pd
import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"
    X_df = train_df.drop(columns=[target_col])
    y_ser = train_df[target_col]

    X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
        X_df, y_ser, test_size=0.2, random_state=42
    )

    data = skrub.var("data", X_train_df)
    X = data.skb.mark_as_X()
    y = skrub.var("target", y_train_ser).skb.mark_as_y()

    vectorizer = skrub.TableVectorizer(
        high_cardinality=skrub.choose_from(
            {
                "gap": skrub.GapEncoder(),
                "minhash": skrub.MinHashEncoder(n_components=8),
            },
            name="high_card_encoder",
        )
    )

    X_vec = X.skb.apply(vectorizer)

    regressor = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )

    pred = X_vec.skb.apply(regressor, y=y)
    learner = pred.skb.make_learner(fitted=True)
    learner.fit({"data": X_train_df, "target": y_train_ser})

    val_pred = learner.predict({"data": X_val_df})
    final_validation_score = mean_squared_error(y_val_ser, val_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    full_data = skrub.var("data", X_df)
    full_X = full_data.skb.mark_as_X()
    full_y = skrub.var("target", y_ser).skb.mark_as_y()
    full_X_vec = full_X.skb.apply(vectorizer)
    full_pred = full_X_vec.skb.apply(regressor, y=full_y)
    full_learner = full_pred.skb.make_learner(fitted=True)
    full_learner.fit({"data": X_df, "target": y_ser})

    test_pred = full_learner.predict({"data": test_df})
    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
