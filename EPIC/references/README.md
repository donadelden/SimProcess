# EPIC Dataset Preprocessor

A utility tool for standardizing and transforming data from the EPIC (Electric Power and Intelligent Control) dataset into a format compatible with the SimProcess framework.

## Overview

The EPIC dataset contains valuable power grid measurements from real systems. This preprocessor converts the EPIC dataset's original format into the standardized format expected by the SimProcess analysis tools, aligning column names and creating derived measurements.

## Usage

```bash
python preprocessor.py <input_csv_file>
```

### Example

```bash
python preprocessor.py EPIC6.csv
```

The output will be saved as `processed_EPIC6.csv` in the same directory.

## Data Transformation

The preprocessor performs the following transformations:

1. Standardizes column names for voltage measurements:

   - `Generation.GIED1.Measurement.V1` → `V1`
   - `Generation.GIED1.Measurement.V2` → `V2`
   - `Generation.GIED1.Measurement.V3` → `V3`

2. Standardizes column names for current measurements:

   - `Generation.GIED1.Measurement.L1_Current` → `C1`
   - `Generation.GIED1.Measurement.L2_Current` → `C2`
   - `Generation.GIED1.Measurement.L3_Current` → `C3`

3. Standardizes column names for power measurements:

   - `Generation.GIED1.Measurement.Real` → `power_real`
   - `Generation.GIED1.Measurement.Reactive` → `power_reactive`
   - `Generation.GIED1.Measurement.Apparent` → `power_apparent`

4. Renames the frequency column:

   - `Generation.GIED1.Measurement.Frequency` → `frequency`

5. Calculates line-to-line voltage differences:

   - `V1_V2` = `V1` - `V2`
   - `V2_V3` = `V2` - `V3`
   - `V1_V3` = `V1` - `V3`

6. Converts and standardizes timestamp information

## Output Format

The output CSV file will contain the following columns:

- `timestamp`: Converted datetime from the original timestamp
- `V1`, `V2`, `V3`: Phase voltages
- `C1`, `C2`, `C3`: Phase currents
- `V1_V2`, `V2_V3`, `V1_V3`: Line-to-line voltage differences
- `frequency`: System frequency
- `power_real`, `power_reactive`, `power_apparent`: Power measurements

## Obtaining the EPIC Dataset

The EPIC dataset should be requested from iTrust, Centre for Research in Cyber Security at the Singapore University of Technology and Design (SUTD).

Dataset request page: [https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)

When using the EPIC dataset in research, please cite the appropriate references provided by SUTD.

## Requirements

- Python 3.6 or higher
- pandas
- numpy

## Notes

- The output filename is automatically generated with a "processed\_" prefix
- The tool validates that the input file exists and has a .csv extension
