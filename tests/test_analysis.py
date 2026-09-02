"""Regression tests pinning the results published in the README.

If a library upgrade or a change to src/analysis.py moves any of these
numbers, the README is now wrong and CI says so.
"""

import unittest

import numpy as np
from sklearn.metrics import confusion_matrix

from src.analysis import FEATURE_NAMES, load_data, run

# Exactly the table in the README's "Model Results" section.
PUBLISHED_RESULTS = {
    "Decision Tree": (0.8387, 0.7083, 0.8500, 0.7727),
    "Random Forest": (0.8065, 0.6818, 0.7500, 0.7143),
    "Logistic Regression": (0.8065, 0.7000, 0.7000, 0.7000),
    "SVM": (0.8065, 0.7000, 0.7000, 0.7000),
    "KNN": (0.7581, 0.6190, 0.6500, 0.6341),
}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.frame = load_data()

    def test_dataset_shape_and_columns(self):
        self.assertEqual(len(self.frame), 310)
        self.assertEqual(list(self.frame.columns), FEATURE_NAMES + ["class"])

    def test_dataset_has_no_missing_values(self):
        self.assertEqual(int(self.frame.isna().sum().sum()), 0)

    def test_class_balance_matches_the_readme(self):
        counts = self.frame["class"].value_counts().to_dict()
        self.assertEqual(counts, {"Abnormal": 210, "Normal": 100})


class ModelResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run().set_index("Model")

    def test_every_model_reproduces_its_published_metrics(self):
        for model, (accuracy, precision, recall, f1) in PUBLISHED_RESULTS.items():
            with self.subTest(model=model):
                row = self.results.loc[model]
                self.assertAlmostEqual(row["Accuracy"], accuracy, places=4)
                self.assertAlmostEqual(row["Precision"], precision, places=4)
                self.assertAlmostEqual(row["Recall"], recall, places=4)
                self.assertAlmostEqual(row["F1-score"], f1, places=4)

    def test_decision_tree_is_the_best_model_by_f1(self):
        self.assertEqual(self.results["F1-score"].idxmax(), "Decision Tree")


class ConfusionMatrixTests(unittest.TestCase):
    def test_decision_tree_confusion_matrix_matches_the_readme(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        frame = load_data()
        features = frame[FEATURE_NAMES]
        target = (frame["class"] == "Normal").astype(int)

        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        scaler = StandardScaler()
        model = DecisionTreeClassifier(random_state=42)
        model.fit(scaler.fit_transform(features_train), target_train)
        predicted = model.predict(scaler.transform(features_test))

        np.testing.assert_array_equal(
            confusion_matrix(target_test, predicted), np.array([[35, 7], [3, 17]])
        )


if __name__ == "__main__":
    unittest.main()
