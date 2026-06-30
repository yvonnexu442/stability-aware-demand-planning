# Raw Data Directory

This directory stores raw public demand planning datasets used by the project.
Raw data files are ignored by Git because they can be large and may be governed
by dataset-specific access rules.

Expected raw data layout:

```text
data/raw/
  favorita/
  m5/
  walmart/
  rossmann/
```

Use `scripts/download_raw_data.py` after configuring Kaggle API credentials.
Most of these benchmark datasets are distributed through Kaggle competitions or
Kaggle-hosted data pages, so downloading may require accepting the relevant
competition rules and placing `kaggle.json` in `~/.kaggle/`.
