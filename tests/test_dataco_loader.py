import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loaders.dataco_loader import (
    build_dataco_daily_product_panel,
    load_dataco_orders,
    profile_dataco_dataset,
)


class DataCoLoaderTest(unittest.TestCase):
    def test_loader_profiles_minimal_dataco_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir)
            orders = pd.DataFrame(
                {
                    "Type": ["DEBIT", "TRANSFER"],
                    "Days for shipping (real)": [3, 5],
                    "Days for shipment (scheduled)": [4, 4],
                    "Delivery Status": ["Advance shipping", "Late delivery"],
                    "Late_delivery_risk": [0, 1],
                    "Category Id": [73, 73],
                    "Category Name": ["Sporting Goods", "Sporting Goods"],
                    "Department Name": ["Fitness", "Fitness"],
                    "Market": ["Pacific Asia", "Pacific Asia"],
                    "Order Country": ["Indonesia", "India"],
                    "order date (DateOrders)": ["1/31/2018 22:56", "1/13/2018 12:27"],
                    "Order Id": [77202, 75939],
                    "Order Item Id": [180517, 179254],
                    "Order Item Quantity": [1, 2],
                    "Sales": [327.75, 655.50],
                    "Order Region": ["Southeast Asia", "South Asia"],
                    "Order Status": ["COMPLETE", "PENDING"],
                    "Product Card Id": [1360, 1360],
                    "Product Name": ["Smart watch", "Smart watch"],
                    "shipping date (DateOrders)": ["2/3/2018 22:56", "1/18/2018 12:27"],
                    "Shipping Mode": ["Standard Class", "Standard Class"],
                    "Customer Email": ["masked", "masked"],
                    "Customer Fname": ["Name", "Name"],
                    "Customer Lname": ["Name", "Name"],
                    "Customer Password": ["masked", "masked"],
                    "Customer Street": ["masked", "masked"],
                    "Customer Zipcode": [725, 725],
                }
            )
            description = pd.DataFrame({"FIELDS": ["Type"], "DESCRIPTION": ["Type of transaction made"]})
            access_logs = pd.DataFrame(
                {
                    "Product": ["Smart watch"],
                    "Category": ["Sporting Goods"],
                    "Date": ["9/1/2017 6:00"],
                    "Month": ["Sep"],
                    "Hour": [6],
                    "Department": ["fitness"],
                    "ip": ["127.0.0.1"],
                    "url": ["/product/smart-watch"],
                }
            )
            orders.to_csv(raw_dir / "DataCoSupplyChainDataset.csv", index=False)
            description.to_csv(raw_dir / "DescriptionDataCoSupplyChain.csv", index=False)
            access_logs.to_csv(raw_dir / "tokenized_access_logs.csv", index=False)

            loaded = load_dataco_orders(raw_dir)
            self.assertIn("shipment_delay_days", loaded.columns)
            self.assertIn("order_date_day", loaded.columns)
            self.assertNotIn("customer_email", loaded.columns)

            panel = build_dataco_daily_product_panel(loaded)
            self.assertEqual(int(panel["demand_units"].sum()), 3)

            profile = profile_dataco_dataset(raw_data_dir=raw_dir)
            self.assertIn("dataco_research_fit", profile)
            self.assertIn("dataco_access_log_summary", profile)
            self.assertIn("Shipment delay and delivery risk", set(profile["dataco_research_fit"]["use_case"]))


if __name__ == "__main__":
    unittest.main()
