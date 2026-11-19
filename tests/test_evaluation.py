import unittest
import numpy as np
from predictml.evaluation import regression_metrics


class MetricTests(unittest.TestCase):
    def test_metric_known_values_and_undefined_cases(self):
        result = regression_metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(result['mae'], 2/3)
        self.assertAlmostEqual(result['mse'], 2/3)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2/3))
        self.assertAlmostEqual(result['wmape'], 100/3)
        self.assertEqual(result['r2'], 0)
        self.assertIsNone(regression_metrics([0, 0], [1, 1])['wmape'])
        self.assertIsNone(regression_metrics([1], [1])['r2'])
        for a, b in [([], []), ([1, 2], [1]), ([float('inf')], [1])]:
            with self.assertRaises(ValueError):
                regression_metrics(a, b)

