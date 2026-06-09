# Crusher Processing Time Dataset

## Overview

This dataset contains 100 observations of processing times (in minutes) for a primary crusher in a mining operation.

The operational context corresponds to a haulage-crushing system where dump trucks transport mineral to the crusher, wait in a queue if necessary, unload their material for processing, and then leave the system empty. The processed mineral is subsequently transferred to downstream operations.

The data exhibit an initial transient period characterized by higher variability, followed by a stable operating regime. This makes the dataset suitable for warm-up period detection and steady-state analysis.

## Column Description

| Column Name       | Data Type | Unit    | Description                                                                        |
| ----------------- | --------- | ------- | ---------------------------------------------------------------------------------- |
| `processing_time` | Float     | Minutes | Time required by the primary crusher to process the material delivered by a truck. |

## Dataset Characteristics

- Number of observations: 100
- Unit: Minutes
- Approximate range: 4–18 minutes
- Estimated transient period: First ~35 observations
- Steady-state mean: Approximately 10.83 minutes
