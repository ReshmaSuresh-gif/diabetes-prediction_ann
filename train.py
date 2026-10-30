import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# load data
df = pd.read_csv("data/raw/diabetes.csv")

# zero is not possible in these columns, so make them missing
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "ANN (64, 32)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=500,
                                  early_stopping=True, n_iter_no_change=20, random_state=42),
    "ANN (16, 8)": MLPClassifier(hidden_layer_sizes=(16, 8), alpha=0.05, max_iter=1000,
                                 early_stopping=True, n_iter_no_change=30, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42),
}

for name in models:
    # fill missing values, scale the data, then train the model
    pipe = Pipeline([("fill", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", models[name])])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    print(name)
    print("  Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("  Precision:", round(precision_score(y_test, pred), 4))
    print("  Recall   :", round(recall_score(y_test, pred), 4))
    print("  F1 score :", round(f1_score(y_test, pred), 4))
    print("  ROC-AUC  :", round(roc_auc_score(y_test, prob), 4))

    # save the final model (ANN 64, 32)
    if name == "ANN (64, 32)":
        joblib.dump(pipe, "ann_model.joblib")
