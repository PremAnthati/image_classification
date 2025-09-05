import os, json, joblib
import numpy as np, pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def main(seed=42, C=10.0, gamma=0.01):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(C=C, gamma=gamma, probability=True)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump({"model": clf, "scaler": scaler}, "models/model.joblib")

    with open("reports/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    print("Accuracy:", acc)

if __name__ == "__main__":
    main()
