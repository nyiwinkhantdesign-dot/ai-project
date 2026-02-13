import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Put kaggle_diabetes.csv next to this script or pass --csv-path."
        ) from e


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"DiabetesPedigreeFunction": "DPF"}).copy(deep=True)

    cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_invalid_zeros] = df[cols_with_invalid_zeros].replace(0, np.nan)

    df["Glucose"] = df["Glucose"].fillna(df["Glucose"].mean())
    df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].mean())
    df["SkinThickness"] = df["SkinThickness"].fillna(df["SkinThickness"].median())
    df["Insulin"] = df["Insulin"].fillna(df["Insulin"].median())
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())

    return df


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 0):
    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=20, random_state=random_state)
    classifier.fit(X_train_scaled, y_train)

    return {
        "scaler": scaler,
        "classifier": classifier,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
    }


def predict_diabetes(
    scaler: StandardScaler,
    classifier: RandomForestClassifier,
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DPF,
    Age,
) -> int:
    x = [[
        int(Pregnancies),
        float(Glucose),
        float(BloodPressure),
        float(SkinThickness),
        float(Insulin),
        float(BMI),
        float(DPF),
        int(Age),
    ]]

    x_scaled = scaler.transform(x)
    return int(classifier.predict(x_scaled)[0])


def evaluate(model_bundle) -> None:
    classifier = model_bundle["classifier"]
    X_test_scaled = model_bundle["X_test_scaled"]
    y_test = model_bundle["y_test"]

    y_pred = classifier.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    acc = round(accuracy_score(y_test, y_pred), 4) * 100

    print("Confusion matrix (test):")
    print(cm)
    print(f"Accuracy on test set: {acc}%")
    print("Classification report (test):")
    print(classification_report(y_test, y_pred))


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="kaggle_diabetes.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args(argv)

    df = load_dataset(args.csv_path)
    df = clean_dataset(df)

    model_bundle = train_model(df, test_size=args.test_size, random_state=args.random_state)

    scores = cross_val_score(
        RandomForestClassifier(n_estimators=20, random_state=args.random_state),
        model_bundle["X_train_scaled"],
        model_bundle["y_train"],
        cv=5,
    )
    avg_acc = round(sum(scores) * 100 / len(scores), 3)
    print(f"Average Accuracy (5-fold CV on train): {avg_acc}%")

    evaluate(model_bundle)

    scaler = model_bundle["scaler"]
    classifier = model_bundle["classifier"]

    examples = [
        (2, 81, 72, 15, 76, 30.1, 0.547, 25),
        (1, 117, 88, 24, 145, 34.5, 0.403, 40),
        (5, 120, 92, 10, 81, 26.1, 0.551, 67),
    ]

    for i, ex in enumerate(examples, start=1):
        pred = predict_diabetes(scaler, classifier, *ex)
        msg = "Oops! You have diabetes." if pred == 1 else "Great! You don't have diabetes."
        print(f"Prediction {i}: {pred} -> {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
