import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# load data
df = pd.read_csv("data/raw/diabetes.csv")

# graph 1: how many patients have diabetes
df["Outcome"].value_counts().sort_index().plot(kind="bar", color=["blue", "orange"])
plt.title("Class distribution (0 = No diabetes, 1 = Diabetes)")
plt.ylabel("Patients")
plt.xticks(rotation=0)
plt.show()

# zero is not possible in these columns, so make them missing
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)

# graph 2: correlation between columns
plt.figure(figsize=(7, 5))
plt.imshow(df.corr(), cmap="coolwarm")
plt.xticks(range(9), df.columns, rotation=90)
plt.yticks(range(9), df.columns)
plt.colorbar()
plt.title("Correlation matrix")
plt.show()

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

accuracy_list = []
probs = {}

for name in models:
    # fill missing values, scale the data, then train the model
    pipe = Pipeline([("fill", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", models[name])])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    probs[name] = prob

    acc = accuracy_score(y_test, pred)
    accuracy_list.append(acc)

    print(name)
    print("  Accuracy :", round(acc, 4))
    print("  Precision:", round(precision_score(y_test, pred), 4))
    print("  Recall   :", round(recall_score(y_test, pred), 4))
    print("  F1 score :", round(f1_score(y_test, pred), 4))
    print("  ROC-AUC  :", round(roc_auc_score(y_test, prob), 4))

    # save the final model (ANN 64, 32)
    if name == "ANN (64, 32)":
        joblib.dump(pipe, "ann_model.joblib")
        ann = pipe.named_steps["model"]
        cm = confusion_matrix(y_test, pred)

# graph 3: training loss of the final ANN
plt.plot(ann.loss_curve_)
plt.title("ANN training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# graph 4: accuracy of all models
plt.bar(list(models.keys()), accuracy_list)
plt.ylim(0, 1)
plt.title("Accuracy of all models")
plt.xticks(rotation=15)
plt.show()

# graph 5: ROC curves
for name in probs:
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves")
plt.legend()
plt.show()

# graph 6: confusion matrix of the final ANN
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (ANN 64, 32)")
plt.show()
