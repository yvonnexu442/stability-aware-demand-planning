import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loaders.m5_loader import load_m5_modeling_table, validate_m5_files


class M5LoaderTest(unittest.TestCase):
    def test_loader_builds_long_panel_with_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            dates = pd.date_range("2011-01-29", periods=40, freq="D")
            calendar = pd.DataFrame(
                {
                    "date": dates,
                    "wm_yr_wk": [11101 + index // 7 for index in range(40)],
                    "weekday": [date.day_name() for date in dates],
                    "wday": [(index % 7) + 1 for index in range(40)],
                    "month": [date.month for date in dates],
                    "year": [date.year for date in dates],
                    "d": ["d_{}".format(index + 1) for index in range(40)],
                    "event_name_1": ["Fixture Event" if index == 10 else None for index in range(40)],
                    "event_type_1": ["Cultural" if index == 10 else None for index in range(40)],
                    "event_name_2": [None for _ in range(40)],
                    "event_type_2": [None for _ in range(40)],
                    "snap_CA": [1 if index % 5 == 0 else 0 for index in range(40)],
                    "snap_TX": [0 for _ in range(40)],
                    "snap_WI": [0 for _ in range(40)],
                }
            )
            sales = pd.DataFrame(
                {
                    "id": ["FOODS_1_001_CA_1_validation", "FOODS_1_002_CA_1_validation"],
                    "item_id": ["FOODS_1_001", "FOODS_1_002"],
                    "dept_id": ["FOODS_1", "FOODS_1"],
                    "cat_id": ["FOODS", "FOODS"],
                    "store_id": ["CA_1", "CA_1"],
                    "state_id": ["CA", "CA"],
                }
            )
            for index in range(40):
                sales["d_{}".format(index + 1)] = [float(index % 4), float((index + 1) % 3)]
            prices = pd.DataFrame(
                {
                    "store_id": ["CA_1", "CA_1"] * 6,
                    "item_id": ["FOODS_1_001", "FOODS_1_002"] * 6,
                    "wm_yr_wk": [11101 + index for index in range(6) for _ in range(2)],
                    "sell_price": [2.0, 3.0] * 6,
                }
            )
            calendar.to_csv(raw_dir / "calendar.csv", index=False)
            sales.to_csv(raw_dir / "sales_train_validation.csv", index=False)
            prices.to_csv(raw_dir / "sell_prices.csv", index=False)

            modeling, quality = load_m5_modeling_table(
                raw_data_dir=raw_dir,
                run_mode="quick",
                max_series=2,
                min_history_length=30,
                min_nonzero_observations=5,
            )

            self.assertEqual(modeling["series_id"].nunique(), 2)
            self.assertIn("sell_price", modeling.columns)
            self.assertIn("snap_active", modeling.columns)
            self.assertIn("demand_lag_7", modeling.columns)
            self.assertIn("zero_demand_rate_28", modeling.columns)
            self.assertGreater(int(quality["selected_series_count"].iloc[0]), 0)

    def test_missing_files_raise_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Missing required M5 files"):
                validate_m5_files(temp_dir)


if __name__ == "__main__":
    unittest.main()
