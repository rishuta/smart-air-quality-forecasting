import unittest
import numpy as np
import pandas as pd
from src.features import TARGET, chronological_masks, make_features


class TemporalIntegrity(unittest.TestCase):
    def setUp(self):
        self.d = pd.DataFrame({TARGET: np.arange(100, dtype=float)},
                              index=pd.date_range("2025-01-01", periods=100, freq="h"))

    def test_future_rows_do_not_change_past_features(self):
        altered = self.d.copy()
        altered.iloc[50:] = 99999
        pd.testing.assert_frame_equal(make_features(self.d).iloc[:50],
                                      make_features(altered).iloc[:50])

    def test_labels_do_not_cross_split_boundaries(self):
        masks = chronological_masks(self.d)
        for left, right in (("train", "validation"), ("validation", "test")):
            self.assertLess((self.d.index[masks[left]] + pd.Timedelta(hours=1)).max(),
                            self.d.index[masks[right]].min())

    def test_missing_targets_excluded(self):
        self.d.iloc[20] = np.nan
        masks = chronological_masks(self.d)
        self.assertFalse(masks["train"][19])
        self.assertFalse(masks["test"][-1])


if __name__ == "__main__":
    unittest.main()
