"""Latih beberapa model, pilih terbaik lewat cross-validation, simpan artefak."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from . import config as C
from .data import load_dataset


def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), C.NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(categories=[C.EXERCISE_LEVELS]), C.CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def candidates():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=C.RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=C.RANDOM_STATE),
    }


def run(df=None, model_path=C.MODEL_PATH, metrics_path=C.METRICS_PATH,
        confusion_path=C.CONFUSION_PATH, verbose=True) -> dict:
    df = load_dataset() if df is None else df
    X, y = df[C.FEATURES], df[C.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    cv_scores = {}
    for name, est in candidates().items():
        s = cross_val_score(build_pipeline(est), X_tr, y_tr, cv=cv, scoring="f1_macro")
        cv_scores[name] = {"mean": float(s.mean()), "std": float(s.std())}
        if verbose:
            print(f"{name:20s} F1-macro CV = {s.mean():.3f} ± {s.std():.3f}")

    best_name = max(cv_scores, key=lambda k: cv_scores[k]["mean"])
    model = build_pipeline(candidates()[best_name]).fit(X_tr, y_tr)

    pred = model.predict(X_te)
    report = classification_report(y_te, pred, labels=C.LABELS, output_dict=True)
    cm = confusion_matrix(y_te, pred, labels=C.LABELS)

    perm = permutation_importance(
        model, X_te, y_te, n_repeats=10, random_state=C.RANDOM_STATE, scoring="f1_macro"
    )
    importance = dict(sorted(zip(C.FEATURES, map(float, perm.importances_mean)),
                             key=lambda kv: -kv[1]))

    metrics = {
        "best_model": best_name,
        "cv_scores": cv_scores,
        "test_accuracy": float(accuracy_score(y_te, pred)),
        "test_f1_macro": float(f1_score(y_te, pred, average="macro")),
        "classification_report": report,
        "permutation_importance": importance,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), C.LABELS)
    ax.set_yticks(range(3), C.LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title(f"Confusion Matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"\nModel terbaik: {best_name}")
        print(f"Test accuracy: {metrics['test_accuracy']:.3f} | F1-macro: {metrics['test_f1_macro']:.3f}")
        print("Fitur paling berpengaruh:", list(importance)[:3])
    return metrics


if __name__ == "__main__":
    run()
