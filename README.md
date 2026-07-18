# Forecasting Financial Inclusion in Ethiopia

## Project Overview

This project develops a financial inclusion forecasting system for Ethiopia. It analyzes historical indicators, infrastructure variables, policies, product launches, and market events to understand the factors influencing financial inclusion.

The project focuses on two Global Findex dimensions:

1. **Access** — Account Ownership Rate
2. **Usage** — Digital Payment Adoption Rate

## Business Objective

The analysis aims to:

- identify the main drivers of financial inclusion in Ethiopia;
- examine relationships between national events and inclusion indicators;
- understand the slowdown in account ownership growth between 2021 and 2024;
- forecast Access and Usage indicators for 2025–2027;
- present the results through an interactive dashboard.

## Repository Structure

```text
ethiopia-fi-forecast/
├── .github/workflows/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── tests/
├── docs/
│   └──data_enrichment_log.md
├── requirements.txt
└── README.md

## Task 1: Data Exploration and Enrichment

Task 1 examines the unified Ethiopia financial inclusion dataset and prepares it for modeling.

The analysis includes:

- Dataset structure and record-type analysis
- Indicator and temporal coverage assessment
- Event and impact-link validation
- Missing-value and documentation analysis
- Collector and collection-date corrections
- Addition of the 2011 Account Ownership baseline
- Export of a validated enriched dataset

### Task 1 Outputs

- `notebooks/task_1_data_exploration_enrichment.ipynb`
- `data/processed/ethiopia_fi_enriched.csv`
- `data/processed/ethiopia_fi_enriched.xlsx`
- `docs/data_enrichment_log.md`

The enriched dataset contains 58 records and supports Account Ownership analysis from 2011 through 2024.
