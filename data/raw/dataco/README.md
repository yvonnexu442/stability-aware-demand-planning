# DataCo Raw Data

This directory is reserved for the DataCo supply chain dataset.

Expected local files:

```text
DataCoSupplyChainDataset.csv
DescriptionDataCoSupplyChain.csv
tokenized_access_logs.csv
```

The raw CSV files are not committed to Git. Use `scripts/profile_dataco.py` to
profile this dataset and write research suitability tables to `outputs/tables/`.
