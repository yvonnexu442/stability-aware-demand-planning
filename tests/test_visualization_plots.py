import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from visualization.plots import (
    apply_paper_style,
    palette_for_strategies,
    save_paper_figure,
    strategy_color,
)


class VisualizationPlotsTest(unittest.TestCase):
    def test_strategy_palette_is_stable(self):
        strategies = ["global_best_model", "feasibility_aware_selector"]
        palette = palette_for_strategies(strategies)

        self.assertEqual(palette["global_best_model"], strategy_color("global_best_model"))
        self.assertEqual(palette["feasibility_aware_selector"], strategy_color("feasibility_aware_selector"))

    def test_save_paper_figure_writes_png_and_pdf(self):
        apply_paper_style()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            save_paper_figure(
                fig,
                png_path=output_dir / "example.png",
                pdf_path=output_dir / "example.pdf",
            )

            self.assertTrue((output_dir / "example.png").exists())
            self.assertTrue((output_dir / "example.pdf").exists())


if __name__ == "__main__":
    unittest.main()
