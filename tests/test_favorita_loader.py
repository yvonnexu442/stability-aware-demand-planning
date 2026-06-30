import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loaders.favorita_loader import load_favorita_modeling_table, validate_favorita_files


class FavoritaLoaderTest(unittest.TestCase):
    def test_loader_builds_modeling_table_and_quality_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            dates = pd.date_range("2017-01-01", periods=120, freq="D")
            train = pd.DataFrame(
                {
                    "id": range(len(dates)),
                    "date": dates,
                    "store_nbr": 1,
                    "family": "GROCERY I",
                    "sales": [0.0 if index % 12 == 0 else float(10 + index % 7) for index in range(len(dates))],
                    "onpromotion": [1 if index % 14 == 0 else 0 for index in range(len(dates))],
                }
            )
            stores = pd.DataFrame(
                {
                    "store_nbr": [1],
                    "city": ["Quito"],
                    "state": ["Pichincha"],
                    "type": ["D"],
                    "cluster": [13],
                }
            )
            oil = pd.DataFrame({"date": dates, "dcoilwtico": [70.0 + index * 0.01 for index in range(len(dates))]})
            holidays = pd.DataFrame(
                {
                    "date": [dates[3]],
                    "type": ["Holiday"],
                    "locale": ["National"],
                    "locale_name": ["Ecuador"],
                    "description": ["Fixture holiday"],
                    "transferred": [False],
                }
            )
            transactions = pd.DataFrame(
                {
                    "date": dates,
                    "store_nbr": 1,
                    "transactions": [1000 + index for index in range(len(dates))],
                }
            )
            train.to_csv(raw_dir / "train.csv", index=False)
            stores.to_csv(raw_dir / "stores.csv", index=False)
            oil.to_csv(raw_dir / "oil.csv", index=False)
            holidays.to_csv(raw_dir / "holidays_events.csv", index=False)
            transactions.to_csv(raw_dir / "transactions.csv", index=False)

            modeling, quality = load_favorita_modeling_table(
                raw_data_dir=raw_dir,
                max_series=1,
                min_history_length=30,
                min_nonzero_observations=20,
            )

            self.assertEqual(modeling["series_id"].nunique(), 1)
            self.assertIn("demand_lag_1", modeling.columns)
            self.assertIn("demand_rolling_mean_28", modeling.columns)
            self.assertIn("demand_ewm_alpha_0_3", modeling.columns)
            self.assertIn("oil_lag_1", modeling.columns)
            self.assertIn("transactions_lag_1", modeling.columns)
            self.assertIn("known_context_available", modeling.columns)
            self.assertEqual(int(quality["series_count"].iloc[0]), 1)
            self.assertGreater(float(quality["promotion_coverage"].iloc[0]), 0.0)

    def test_missing_files_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Missing required Favorita files"):
                validate_favorita_files(temp_dir)


if __name__ == "__main__":
    unittest.main()
