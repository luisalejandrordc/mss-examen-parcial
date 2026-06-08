# Mining Simulation Data Dictionary

## Dataset Overview

This dataset contains simulated observations of dump truck arrivals and mineral processing operations in a mining crushing plant. Each row represents a single truck and its associated operational parameters.

## Columns

| Column Name         | Data Type | Unit    | Description                                                                   |
| ------------------- | --------- | ------- | ----------------------------------------------------------------------------- |
| `truck_id`          | Integer   | N/A     | Unique identifier assigned to each truck in the simulation.                   |
| `interarrival_time` | Float     | Minutes | Time elapsed between the arrival of the current truck and the previous truck. |
| `crushing_time`     | Float     | Minutes | Time required for the crusher to process the truck's load.                    |
| `mineral_load`      | Float     | Tonnes  | Amount of mineral transported by the truck.                                   |

## Data Generation Assumptions

### Interarrival Time

- Distribution: Exponential
- Mean: 15 minutes
- Purpose: Models the random arrival pattern of dump trucks.

### Crushing Time

- Distribution: Normal
- Mean: 12 minutes
- Standard Deviation: 2 minutes
- Purpose: Models the variability in crusher processing times.

### Mineral Load

- Distribution: Uniform
- Minimum: 30 tonnes
- Maximum: 40 tonnes
- Purpose: Models variation in the amount of mineral transported by each truck.

## Example Record

| truck_id | interarrival_time | crushing_time | mineral_load |
| -------- | ----------------- | ------------- | ------------ |
| 1        | 14.8              | 11.5          | 36.2         |
