# Data Enrichment Log

## Project

Ethiopia Financial Inclusion Forecasting

## Analyst

Bilen M. Gebremariam

## Purpose

This document records the cleaning, correction, and enrichment decisions applied to the starter financial inclusion dataset.

## Original Dataset

The original workbook contained 57 records:

- 30 observations
- 10 events
- 14 impact links
- 3 targets

The dataset used a unified schema that stored measured observations, policy targets, events, and modeled impact relationships.

## Cleaning Actions

### Collector Field Correction

Forty-three records contained the value `2025-01-20` in the `collected_by` field. This value was interpreted as a misplaced collection date.

The date was moved to the `collection_date` field, and the collector name was standardized as:

`Bilen M. Gebremariam`

### Placeholder Removal

Fourteen impact-link records contained the placeholder collector name:

`Example_Trainee`

The placeholder was replaced with:

`Bilen M. Gebremariam`

### Date Standardization

The following columns were converted to datetime format where applicable:

- observation_date
- period_start
- period_end
- collection_date

## Enrichment Action

The original dataset did not contain the 2011 Account Ownership baseline required for analysis from 2011 to 2024.

The following observation was added:

- Record ID: REC_0058
- Record type: observation
- Pillar: ACCESS
- Indicator: Account Ownership Rate
- Indicator code: ACC_OWNERSHIP
- Value: 14%
- Observation date: 2011-12-31
- Gender: all
- Geography: Ethiopia
- Source: World Bank Global Findex 2011
- Confidence: high

## Modeling Considerations

The national Account Ownership series now contains five observations:

- 2011: 14%
- 2014: 22%
- 2017: 35%
- 2021: 46%
- 2024: 49%

Gender-disaggregated observations were retained but excluded from the national trend series.

Registered mobile-money accounts and survey-reported adult account ownership represent different concepts and should not be treated as directly equivalent.

Events remain neutral records without direct pillar assignments. Their expected effects are represented through impact-link records.

## Final Dataset

The enriched dataset contains 58 records.

Validation confirmed:

- No duplicate record IDs
- No missing record IDs
- No missing record types
- No missing collector names
- No missing collection dates
- No remaining placeholder collector names
- Five national Account Ownership observations
- Account Ownership coverage from 2011 through 2024

## Exported Files

- `data/processed/ethiopia_fi_enriched.csv`
- `data/processed/ethiopia_fi_enriched.xlsx`