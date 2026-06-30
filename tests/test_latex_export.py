import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reporting.latex_export import (
    dataframe_to_latex_table,
    export_dataframe_to_latex,
    latex_figure_snippet,
    latex_table_snippet,
)


class LatexExportTest(unittest.TestCase):
    def test_dataframe_export_writes_csv_and_latex(self):
        data = pd.DataFrame(
            {
                "model_name": ["baseline"],
                "forecast_error": [1.23456],
                "execution_penalty": [0.5],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "forecast_metrics_table.tex"
            paths = export_dataframe_to_latex(
                data=data,
                tex_path=output_path,
                caption="Forecast Metrics Table",
                label="tab:forecast-metrics",
                numeric_precision=2,
                column_renames={"model_name": "Model Name"},
            )

            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["tex"].exists())
            latex_text = paths["tex"].read_text(encoding="utf-8")
            self.assertIn("\\toprule", latex_text)
            self.assertIn("Forecast Metrics Table", latex_text)
            self.assertIn("tab:forecast-metrics", latex_text)
            self.assertIn("1.23", latex_text)

    def test_snippet_helpers(self):
        table_snippet = latex_table_snippet("tables/forecast_metrics_table.tex")
        figure_snippet = latex_figure_snippet(
            "figures/accuracy_vs_inventory_cost.pdf",
            caption="Accuracy Versus Inventory Cost",
            label="fig:accuracy-inventory",
        )
        self.assertEqual(table_snippet, "\\input{tables/forecast_metrics_table.tex}\n")
        self.assertIn("\\includegraphics", figure_snippet)
        self.assertIn("accuracy_vs_inventory_cost.pdf", figure_snippet)

    def test_dataframe_to_latex_requires_booktabs(self):
        data = pd.DataFrame({"metric": ["mae"], "value": [1.0]})
        with self.assertRaises(ValueError):
            dataframe_to_latex_table(data, caption="Metrics", label="tab:metrics", booktabs=False)


if __name__ == "__main__":
    unittest.main()
