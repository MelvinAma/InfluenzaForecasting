# Data Pipeline

This directory contains the converted data used for both the US and Swedish influenza forecasting experiments, formatted for the EpiGNN and ColaGNN loaders.

Shapes and summary statistics below match the numbers reported in the thesis Methodology and Implementation chapters. The earlier "Phase 1" version of this README referenced an older data snapshot with fewer weeks; those values have been superseded by the most recent fetches and the numbers below are the ones used in the thesis.

## Generated files

### US data (51 jurisdictions: 50 states + DC)

- `state_flu_admissions.txt` — Time series matrix, 291 weeks x 51 jurisdictions
  - Source: CDC NHSN Weekly Hospital Respiratory Data
  - Metric: Total Influenza Admissions (weekly counts)
  - Date range: August 2020 to February 2026
  - Format: comma-separated floats, no header
  - Zero values: 17.8%, concentrated in the early COVID period (Aug to Oct 2020)
  - Season use: the checked-in US configs exclude the suppressed 2020/21 season and the incomplete 2025/26 season. The effective evaluation window is 2021/22 to 2024/25.

- `state-adj-51.txt` — Adjacency matrix, 51 x 51
  - Binary: 1 = shared land border, 0 = no border
  - 218 nonzero entries (109 undirected edges, average degree 4.3)
  - Symmetric, zero diagonal
  - Isolated nodes: Alaska, Hawaii
  - DC is connected to Maryland and Virginia

- `state_index.csv` — State order reference (column mapping)
- `date_index.csv` — Week-to-date mapping

### Swedish data (21 counties / laen)

- `sweden_flu_cases.txt` — Time series matrix, 538 weeks x 21 regions
  - Source: Folkhalsomyndigheten (Swedish Public Health Agency)
  - Metric: laboratory-confirmed influenza cases (weekly counts)
  - Date range: 2015 week 40 to 2026 week 3 (about 10 influenza seasons)
  - Format: comma-separated floats, no header
  - Zero values: 46.8%, reflecting both lower baseline surveillance in smaller counties and genuine low-activity weeks

- `sweden_flu_rates_per100k.txt` — Same underlying data, rescaled to weekly rate per 100,000 population per county. Kept alongside the absolute count matrix for exploratory use. The thesis experiments use the absolute counts.

- `sweden-adj-21.txt` — Adjacency matrix, 21 x 21
  - Binary: 1 = shared land border, 0 = no border
  - 74 nonzero entries (37 undirected edges, average degree 3.5)
  - Symmetric, zero diagonal
  - Isolated node: Gotland

- `sweden_region_index.csv` — Region order reference (column mapping)
- `sweden_week_index.csv` — Week-to-date mapping

## Converters

### `nhsn_converter.py`

Converts the CDC NHSN semicolon-delimited CSV to the EpiGNN / ColaGNN matrix format.

- Input: `Data/Weekly_Hospital_Respiratory_Data_*.csv`
- Filters to 51 US jurisdictions (excludes territories and the US national aggregate)
- Validates required columns, detects duplicate state-week pairs, confirms output dimensions

### `sweden_converter.py`

Converts the Folkhalsomyndigheten wide-format CSV to the same matrix format.

- Input: `Data/Swedish_Folkhalsomyndigheten.csv`
- Transposes from `[regions x weeks]` to `[weeks x regions]`
- ISO-8859-1 encoding for Swedish characters (A, O, A-ring)
- Parses week columns of the form `Totalt influensa Antal fall YYYY v WW`

### `adjacency_builder.py`

Generates the geographic adjacency matrices from hand-coded neighbor lists.

- Verifies that every named neighbor exists in the entity list
- Confirms symmetry and zero diagonal
- Reports edge counts and average degree

### `validate_formats.py`

Validates the generated files against the format expected by the upstream EpiGNN and ColaGNN loaders. Checks matrix dimensions, numeric values, adjacency symmetry, and reports data statistics.

## References

- US data: [CDC NHSN Weekly Hospital Respiratory Data](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/mpgq-jmmr)
- Swedish data: [Folkhalsomyndigheten influenza surveillance](https://www.folkhalsomyndigheten.se/folkhalsorapportering-statistik/)
- EpiGNN paper: Xie et al. (2022), IJCAI
- ColaGNN paper: Deng et al. (2020), CIKM
