# System Flow

The repository models the path from public demand data to operational planning evidence.

```mermaid
flowchart TD
    A["Public Demand Dataset"] --> B["Data Quality Check"]
    B --> C["Chronological Train / Validation / Test Split"]
    C --> D["Feature Engineering"]
    D --> E["Candidate Forecast Models"]
    E --> F["Candidate Forecasts"]
    F --> G["Decision Layer"]
    G --> H["Planning Signal Conversion"]
    H --> I["Execution Capability Model"]
    I --> J["Policy / Context Drift Scenarios"]
    J --> K["Operational Planning Simulator"]
    K --> L["Multi-objective Evaluation"]
    L --> M["Tables / Figures / Paper Claims"]
```

The decision layer is the research focus. It converts candidate forecasts into planning signals while accounting for model switching, planning signal volatility, and execution capacity.
