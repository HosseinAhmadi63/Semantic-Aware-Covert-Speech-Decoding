import unittest

import numpy as np

from semantic_speech.metrics import (
    bca_interval,
    classification_metrics,
    evaluate_fold,
    exact_sign_flip,
    fit_temperature,
    holm_adjust,
    knn_accuracy,
    probabilities,
    reliability_curve,
    semantic_retrieval,
)


class MetricsTests(unittest.TestCase):
    def test_probability_stability(self):
        actual = probabilities([[1000, 1001], [-1000, -1001]])
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual.sum(axis=1), 1.0)
        self.assertGreater(actual[0, 1], actual[0, 0])

    def test_bin_boundaries_and_empty_bins(self):
        values = np.array([[0.5, 0.5], [0.8, 0.2], [0.0, 1.0]])
        curve = reliability_curve(values, np.array([0, 1, 1]), bins=10)
        self.assertEqual(sum(row["count"] for row in curve), 3)
        self.assertEqual(curve[5]["count"], 1)
        self.assertEqual(curve[8]["count"], 1)
        self.assertEqual(curve[9]["count"], 1)
        self.assertIsNone(curve[0]["accuracy"])
        self.assertEqual(curve[9]["accuracy"], 1.0)
        metrics = classification_metrics(np.log(np.maximum(values, 1e-100)), np.array([0, 1, 1]))
        self.assertAlmostEqual(metrics["ece"], (0.5 + 0.8) / 3)
        self.assertAlmostEqual(metrics["brier_sum"], 2 * metrics["brier_mean"])

    def test_rank_metrics(self):
        scores = np.array([[4, 3, 2, 1], [4, 3, 2, 1], [4, 3, 2, 1]])
        metrics = classification_metrics(scores, np.array([0, 1, 3]))
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["top3"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], (1 + 1 / 2 + 1 / 4) / 3)

    def test_semantic_cosine_normalization(self):
        prototypes = np.eye(4)
        embeddings = np.array([[100, 0, 0, 0], [0, 0, 0, 0.1]])
        metrics = semantic_retrieval(embeddings, prototypes * 9, np.array([0, 3]))
        self.assertEqual(metrics["top1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_knn_uses_training_bank(self):
        train_x = np.array([[1, 0], [0, 1], [-1, 0]])
        train_y = np.array([0, 1, 1])
        test_x = np.array([[0.9, 0.1], [0, 1]])
        test_y = np.array([0, 1])
        self.assertEqual(knn_accuracy(train_x, train_y, test_x, test_y, k=1), 1.0)
        self.assertEqual(knn_accuracy(train_x, train_y, test_x, test_y, k=99), 0.5)

    def test_exact_sign_flip_known_distribution(self):
        self.assertAlmostEqual(exact_sign_flip([1, 1, 1], alternative="greater"), 1 / 8)
        self.assertAlmostEqual(exact_sign_flip([1, 1, 1]), 2 / 8)
        self.assertAlmostEqual(exact_sign_flip([1e-15, 1e-15, 1e-15]), 2 / 8)
        self.assertEqual(exact_sign_flip([0, 0, 0]), 1.0)
        self.assertEqual(exact_sign_flip([-1, -1, -1], alternative="greater"), 1.0)

    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03, 0.9]), [0.09, 0.04, 0.09, 0.9])
        np.testing.assert_allclose(holm_adjust([0.01, 0.01]), [0.02, 0.02])

    def test_bca_constant_interval(self):
        self.assertEqual(bca_interval([0.3] * 10), (0.3, 0.3))
        self.assertEqual(bca_interval([0.7]), (0.7, 0.7))
        lower, upper = bca_interval([0.1, 0.2, 0.4, 0.6, 0.8], n_resamples=2000)
        self.assertLess(lower, 0.42)
        self.assertGreater(upper, 0.42)

    def test_temperature_reduces_overconfidence(self):
        logits = np.array([[8, 0], [8, 0], [0, 8], [0, 8]], dtype=float)
        labels = np.array([0, 1, 1, 0])
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 1)
        calibrated = probabilities(logits, temperature)
        original = probabilities(logits)
        rows = np.arange(4)
        self.assertLess(-np.log(calibrated[rows, labels]).mean(), -np.log(original[rows, labels]).mean())

    def test_fold_calibration_ignores_covert_outcomes(self):
        logits = np.array([[3, 0], [3, 0], [0, 3], [0, 3]], dtype=float)
        labels = np.array([0, 1, 1, 0])
        embedding = np.eye(2)[labels]
        result = {
            "overt_val_logits": logits,
            "covert_test_logits": logits.copy(),
            "overt_val_y": labels,
            "covert_test_y": labels.copy(),
            "covert_train_y": labels,
            "covert_val_y": labels,
            "overt_val_semantic": embedding,
            "covert_test_semantic": embedding.copy(),
            "covert_train_embedding": embedding,
            "covert_val_embedding": embedding,
            "covert_test_embedding": embedding,
        }
        first = evaluate_fold(result, np.eye(2))
        result["covert_test_logits"] = logits * 100
        result["covert_test_y"] = 1 - labels
        second = evaluate_fold(result, np.eye(2))
        self.assertEqual(first["temperature"], second["temperature"])
        self.assertEqual(first["overt_validation"], second["overt_validation"])
        without_semantics = evaluate_fold(result, None)
        self.assertIsNone(without_semantics["overt_semantic"])
        self.assertIsNone(without_semantics["covert_semantic"])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([[1, np.nan]], np.array([0]))
        with self.assertRaises(ValueError):
            classification_metrics([[1, 2]], np.array([2]))
        with self.assertRaises(ValueError):
            probabilities([[1, 2]], temperature=0)
        with self.assertRaises(ValueError):
            reliability_curve([[0.8, 0.8]], np.array([0]))

    def test_m1_semantic_metrics_use_intentions(self):
        labels = np.array([0, 4, 1, 5])
        embedding = np.eye(4)[labels % 4]
        logits = 3 * np.eye(8)[labels]
        result = {
            "variant": "M1",
            "overt_val_logits": logits,
            "covert_test_logits": logits,
            "overt_val_y": labels,
            "covert_test_y": labels,
            "covert_train_y": labels,
            "covert_val_y": labels,
            "overt_val_semantic": embedding,
            "covert_test_semantic": embedding,
            "covert_train_embedding": embedding,
            "covert_val_embedding": embedding,
            "covert_test_embedding": embedding,
        }
        metrics = evaluate_fold(result, np.concatenate([np.eye(4), np.eye(4)]))
        self.assertEqual(metrics["semantic_label_space"], "intention")
        self.assertEqual(metrics["overt_semantic"]["top1"], 1.0)
        self.assertEqual(metrics["covert_semantic"]["mrr"], 1.0)
        self.assertEqual(metrics["covert_test"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
