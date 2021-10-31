import unittest
import torch
from kernprior import metrics


class TestMetrics(unittest.TestCase):
    def test_metrics_preprocess(self):
        target = 0.1 * torch.randn(320, 320) - 2
        pred = 3 * torch.randn(320, 320) + 5

        target_to_target, pred_to_target = metrics.metrics_preprocess(target, pred, normalize_to='target')
        target_to_pred, pred_to_pred = metrics.metrics_preprocess(target, pred, normalize_to='pred')

        def round_2(x):
            return round(float(x), 2)

        self.assertTrue(round_2(target.mean()) == round_2(target_to_target.mean()) == round_2(pred_to_target.mean()))
        self.assertTrue(round_2(target.std()) == round_2(target_to_target.std()) == round_2(pred_to_target.std()))

        self.assertTrue(round_2(pred.mean()) == round_2(target_to_pred.mean()) == round_2(pred_to_pred.mean()))
        self.assertTrue(round_2(pred.std()) == round_2(target_to_pred.std()) == round_2(pred_to_pred.std()))


if __name__ == '__main__':
    unittest.main()
