"""Reproducible end-to-end analysis for the Lower Back Pain Prediction project.

Loads the biomechanical spine dataset, cleans it, regenerates every figure in
`images/`, trains five classifiers, and prints the comparison table reported in
the README.

Run from the repository root:

    python src/analysis.py              # print the metrics
    python src/analysis.py --figures    # also regenerate images/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Dataset_spine.csv"
IMAGES_DIR = PROJECT_ROOT / "images"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# The published dataset ships generic Col1..Col12 headers. This is the mapping
# documented by the Kaggle source, in column order.
FEATURE_NAMES = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "degree_spondylolisthesis",
    "pelvic_slope",
    "direct_tilt",
    "thoracic_slope",
    "cervical_tilt",
    "sacrum_angle",
    "scoliosis_slope",
]


def load_data() -> pd.DataFrame:
    """Read the raw CSV, drop the descriptive trailing column, rename headers."""
    frame = pd.read_csv(DATA_PATH)

    # The CSV has a trailing comma on every row, which pandas reads as an
    # all-empty "Unnamed: 13" column. The source file also uses it to carry a
    # free-text description rather than patient data.
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]

    frame = frame.rename(
        columns={f"Col{index}": name for index, name in enumerate(FEATURE_NAMES, start=1)}
        | {"Class_att": "class"}
    )
    return frame


def plot_class_distribution(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=frame, x="class", hue="class", palette="Set2", legend=False)
    plt.title("Class distribution")
    plt.xlabel("")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "class_distribution.png", dpi=150)
    plt.close()


def plot_correlation_matrix(frame: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        frame[FEATURE_NAMES].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation matrix of biomechanical features")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_matrix.png", dpi=150)
    plt.close()


def plot_feature_boxplots(frame: pd.DataFrame) -> None:
    for feature in FEATURE_NAMES:
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            data=frame, x="class", y=feature, hue="class", palette="Set2", legend=False
        )
        plt.title(f"{feature.replace('_', ' ').capitalize()} by class")
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"boxplot_{feature}.png", dpi=150)
        plt.close()


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
    }


def run(write_figures: bool = False) -> pd.DataFrame:
    """Run the full pipeline and return the model comparison table."""
    frame = load_data()
    print(f"Rows: {len(frame)}   Columns: {frame.shape[1]}")
    print(f"Missing values: {int(frame.isna().sum().sum())}")
    print(frame["class"].value_counts().to_string(), end="\n\n")

    if write_figures:
        IMAGES_DIR.mkdir(exist_ok=True)
        plot_class_distribution(frame)
        plot_correlation_matrix(frame)
        plot_feature_boxplots(frame)

    features = frame[FEATURE_NAMES]
    # Abnormal -> 0, Normal -> 1, as documented in the README.
    target = (frame["class"] == "Normal").astype(int)

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    rows = []
    predictions_by_model = {}
    for name, model in build_models().items():
        model.fit(features_train_scaled, target_train)
        predicted = model.predict(features_test_scaled)
        predictions_by_model[name] = (model, predicted)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(target_test, predicted),
                "Precision": precision_score(target_test, predicted, zero_division=0),
                "Recall": recall_score(target_test, predicted, zero_division=0),
                "F1-score": f1_score(target_test, predicted, zero_division=0),
            }
        )

    results = (
        pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if write_figures:
        ordered = results.set_index("Model").loc[list(build_models())].reset_index()
        plt.figure(figsize=(8, 5))
        plt.bar(ordered["Model"], ordered["F1-score"], color="#2e7ebb")
        plt.title("Model Comparison Based on F1 Score")
        plt.xlabel("Model")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "model_comparison.png", dpi=150)
        plt.close()

    best_name = results.loc[0, "Model"]
    best_model, best_predictions = predictions_by_model[best_name]
    print(f"\nBest model by F1-score: {best_name}")

    matrix = confusion_matrix(target_test, best_predictions)
    print("Confusion matrix (rows: true Abnormal/Normal, cols: predicted)")
    print(matrix)

    if not write_figures:
        return results

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Abnormal", "Normal"],
        yticklabels=["Abnormal", "Normal"],
        cbar=False,
    )
    plt.title(f"Confusion matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        importance = (
            pd.Series(best_model.feature_importances_, index=FEATURE_NAMES)
            .sort_values(ascending=True)
        )
        plt.figure(figsize=(8, 6))
        importance.plot(kind="barh", color="#4C9F70")
        plt.title(f"Feature importance — {best_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "feature_importance.png", dpi=150)
        plt.close()
        print("\nFeature importance (descending)")
        print(importance.sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        action="store_true",
        help="regenerate the PNG files in images/ (off by default)",
    )
    run(write_figures=parser.parse_args().figures)


if __name__ == "__main__":
    main()
