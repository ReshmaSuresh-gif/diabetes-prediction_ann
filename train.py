"""Train and evaluate Logistic Regression, two ANNs and Random Forest."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import ROOT, load_raw, clean, FEATURES, TARGET

FIG = ROOT / "reports" / "figures"
SEED = 42


def make_pipeline(model):
    pre = ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), FEATURES)])
    return Pipeline([("prep", pre), ("model", model)])


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "ANN (64, 32)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                      solver="adam", alpha=0.001, learning_rate_init=0.001,
                                      max_iter=500, early_stopping=True, validation_fraction=0.1,
                                      n_iter_no_change=20, random_state=SEED),
        "ANN (16, 8) regularised": MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                                      solver="adam", alpha=0.05, learning_rate_init=0.001,
                                      max_iter=1000, early_stopping=True, validation_fraction=0.1,
                                      n_iter_no_change=30, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                random_state=SEED, n_jobs=-1),
    }


def main():
    df = clean(load_raw())
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print("train", Xtr.shape, "test", Xte.shape, "| positives train/test:", ytr.sum(), yte.sum())

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted, probs, curves = [], {}, {}, {}
    for name, m in models().items():
        pipe = make_pipeline(m)
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(yte, pred),
                     "Precision": precision_score(yte, pred),
                     "Recall": recall_score(yte, pred),
                     "F1": f1_score(yte, pred),
                     "ROC-AUC": roc_auc_score(yte, p),
                     "CV ROC-AUC (mean)": cv_auc.mean(), "CV ROC-AUC (std)": cv_auc.std(),
                     "TN/FP/FN/TP": "/".join(map(str, confusion_matrix(yte, pred).ravel()))})
        fitted[name], probs[name] = pipe, p
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

    # final model: best mean cross-validated ROC-AUC among the ANN variants (selection uses training data only)
    ann_names = [n for n in fitted if n.startswith("ANN")]
    best_ann = res[res.Model.isin(ann_names)].sort_values("CV ROC-AUC (mean)", ascending=False).iloc[0].Model
    overall = res.sort_values("CV ROC-AUC (mean)", ascending=False).iloc[0].Model
    print("Best ANN by CV:", best_ann, "| best overall by CV:", overall)
    joblib.dump(fitted[best_ann], ROOT / "models" / "ann_pipeline.joblib")
    json.dump({"final_model": best_ann, "best_overall_cv": overall,
               "feature_medians": X.median().round(2).to_dict(),
               "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES}},
              open(ROOT / "models" / "meta.json", "w"), indent=2)

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cols = ["#1f4e79", "#2e86c1", "#e67e22", "#7f8c8d"]
    # class distribution
    fig, ax = plt.subplots(figsize=(4.5, 3.2)); vc = y.value_counts().sort_index()
    ax.bar(["Non-diabetic (0)", "Diabetic (1)"], vc.values, color=[cols[1], cols[2]])
    for i, v in enumerate(vc.values): ax.text(i, v + 5, str(v), ha="center")
    ax.set_ylabel("Patients"); ax.set_title("Class distribution"); fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=200); plt.close(fig)
    # correlation heatmap
    fig, ax = plt.subplots(figsize=(6.2, 5)); c = df.corr(numeric_only=True)
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns)
    for i in range(len(c)):
        for j in range(len(c)): ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, shrink=0.8); ax.set_title("Correlation matrix"); fig.tight_layout()
    fig.savefig(FIG / "correlation.png", dpi=200); plt.close(fig)
    # missing values
    fig, ax = plt.subplots(figsize=(5.5, 3.2)); mv = df[FEATURES].isna().sum(); mv = mv[mv > 0]
    ax.barh(mv.index, mv.values, color=cols[0]); ax.set_xlabel("Missing (recorded as 0) values")
    ax.set_title("Missing values after cleaning"); fig.tight_layout()
    fig.savefig(FIG / "missing_values.png", dpi=200); plt.close(fig)
    # loss curve of the final ANN
    mlp = fitted[best_ann].named_steps["model"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(mlp.loss_curve_, label="Training loss", color=cols[0])
    ax.plot(np.arange(len(mlp.validation_scores_)), mlp.validation_scores_, label="Validation accuracy", color=cols[2])
    ax.set_xlabel("Epoch"); ax.set_title(f"Training history: {best_ann}"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "loss_curve.png", dpi=200); plt.close(fig)
    # ROC
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for (n, p), col in zip(probs.items(), cols):
        fpr, tpr, _ = roc_curve(yte, p); ax.plot(fpr, tpr, label=f"{n} (AUC {roc_auc_score(yte, p):.3f})", color=col)
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)"); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "roc.png", dpi=200); plt.close(fig)
    # confusion matrices
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, (n, p) in zip(axs, probs.items()):
        cm = confusion_matrix(yte, (p >= .5).astype(int)); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(n, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "confusion.png", dpi=200); plt.close(fig)
    # metric comparison
    fig, ax = plt.subplots(figsize=(7, 3.6)); metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    w = 0.2
    for k, (n, col) in enumerate(zip(res.Model, cols)):
        ax.bar(np.arange(5) + k * w, res.loc[k, metrics].astype(float), w, label=n, color=col)
    ax.set_xticks(np.arange(5) + 1.5 * w); ax.set_xticklabels(metrics); ax.set_ylim(0, 1)
    ax.set_title("Test-set metric comparison"); ax.legend(fontsize=7, ncol=2, loc="lower center"); fig.tight_layout()
    fig.savefig(FIG / "metric_comparison.png", dpi=200); plt.close(fig)

    # permutation importance for final model
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(fitted[best_ann], Xte, yte, scoring="roc_auc", n_repeats=30, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    imp.to_csv(ROOT / "reports" / "permutation_importance.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.6)); ax.barh(imp.index, imp.values, color=cols[0])
    ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("Permutation importance (final ANN)"); fig.tight_layout()
    fig.savefig(FIG / "importance.png", dpi=200); plt.close(fig)
    print(imp.round(4).to_string())
    print("final ANN n_iter:", mlp.n_iter_, "layers:", mlp.hidden_layer_sizes, "best_val:", round(mlp.best_validation_score_, 4))


if __name__ == "__main__":
    main()
