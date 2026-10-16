"""Testes básicos do motor sem retreinar o Ridge reduzido."""
import unittest

import numpy as np

from src.recommendations_engine import RecommendationsEngine


DEMO = {
    "campaign_id": "DEMO-MANUAL-001",
    "platform": "Meta",
    "category": "E-commerce",
    "is_retargeting": False,
    "video_duration_s": 16,
    "format": "vertical",
    "has_subtitle": False,
    "has_cta": False,
    "has_hook": False,
    "has_face": False,
    "text_density": "low",
}


class RecommendationsEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationsEngine()

    def test_split_and_contextual_isolation(self):
        self.assertEqual(len(self.engine.X_train), 400)
        self.assertEqual(len(self.engine.X_test), 100)
        self.assertTrue(set(self.engine.X_train.index).isdisjoint(self.engine.X_test.index))
        self.assertEqual(set(self.engine.df_evidencias.index), set(self.engine.X_train.index))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.engine.preparar_campanha({**DEMO, "has_face": "false"})
        with self.assertRaises(ValueError):
            self.engine.preparar_campanha({**DEMO, "video_duration_s": -2})

    def test_inference_ignores_outcome_columns(self):
        original = self.engine.preparar_campanha(DEMO)
        extra = self.engine.preparar_campanha({
            **DEMO, "klike_score": 100, "ctr": 0.99, "revenue": 999999
        })
        self.assertTrue(original.equals(extra))
        self.assertTrue(np.allclose(
            self.engine.modelo_ridge.predict(original),
            self.engine.modelo_ridge.predict(extra)
        ))

    def test_demo_has_three_ranked_recommendations(self):
        result = self.engine.recomendar_campanha(DEMO, n_bootstrap=2000)
        self.assertEqual(result["total_recomendacoes"], 3)
        self.assertEqual(
            [x["feature"] for x in result["recomendacoes"]],
            ["has_hook", "has_face", "has_subtitle"],
        )
        deltas = [r["delta_ridge"] for r in result["recomendacoes"]]
        self.assertEqual(deltas, sorted(deltas, reverse=True))


if __name__ == "__main__":
    unittest.main()
