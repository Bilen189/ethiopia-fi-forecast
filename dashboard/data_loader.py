"""
Data loading and preparation utilities for the Ethiopia Financial
Inclusion Forecasting Dashboard.

The functions in this module are intentionally defensive because
dashboard deployments may run from different working directories and
may encounter partially available model outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# -------------------------------------------------------------------
# Candidate input files
# -------------------------------------------------------------------

ENRICHED_DATA_CANDIDATES = [
    PROCESSED_DATA_DIR / "ethiopia_fi_enriched.csv",
    PROCESSED_DATA_DIR / "ethiopia_fi_unified_data_enriched.csv",
    DATA_DIR / "raw" / "ethiopia_fi_unified_data.csv",
]

FORECAST_SCENARIO_CANDIDATES = [
    MODELS_DIR / "task4_forecast_scenarios_2025_2027.csv",
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "task4_forecast_scenarios_2025_2027.csv",
]

BASELINE_FORECAST_CANDIDATES = [
    MODELS_DIR / "task4_baseline_forecasts_2025_2027.csv",
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "task4_baseline_forecasts_2025_2027.csv",
]

EVENT_SUMMARY_CANDIDATES = [
    MODELS_DIR / "event_indicator_association_summary.csv",
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "event_indicator_association_summary.csv",
]

EVENT_MATRIX_CANDIDATES = [
    MODELS_DIR / "event_indicator_association_matrix.csv",
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "event_indicator_association_matrix.csv",
]

TARGET_PROGRESS_CANDIDATES = [
    MODELS_DIR / "task4_access_target_progress.csv",
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "task4_access_target_progress.csv",
]

LARGEST_EVENT_IMPACTS_CANDIDATES = [
    MODELS_DIR / "task4_largest_event_impacts.csv",
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "task4_largest_event_impacts.csv",
]


# -------------------------------------------------------------------
# Generic helper functions
# -------------------------------------------------------------------

def first_existing_path(
    candidates: Iterable[Path],
) -> Path | None:
    """
    Return the first existing file from a collection of paths.

    Parameters
    ----------
    candidates:
        Candidate filesystem paths.

    Returns
    -------
    Path or None
        First existing path, or None when no candidate exists.
    """
    for candidate in candidates:
        candidate = Path(candidate)

        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def safe_read_csv(
    candidates: Iterable[Path],
    required: bool = False,
    dataset_name: str = "dataset",
) -> tuple[pd.DataFrame, Path | None]:
    """
    Read the first available CSV from candidate paths.

    Parameters
    ----------
    candidates:
        Possible CSV locations.
    required:
        Whether absence of the CSV should raise an exception.
    dataset_name:
        Human-readable dataset description.

    Returns
    -------
    tuple
        DataFrame and the path used.
    """
    selected_path = first_existing_path(candidates)

    if selected_path is None:
        if required:
            checked_paths = "\n".join(
                f"- {Path(path)}"
                for path in candidates
            )

            raise FileNotFoundError(
                f"Could not locate the required {dataset_name}.\n"
                f"Checked:\n{checked_paths}"
            )

        return pd.DataFrame(), None

    try:
        dataframe = pd.read_csv(selected_path)
    except pd.errors.EmptyDataError:
        if required:
            raise ValueError(
                f"The required {dataset_name} is empty: "
                f"{selected_path}"
            )

        return pd.DataFrame(), selected_path
    except Exception as error:
        if required:
            raise RuntimeError(
                f"Unable to read {dataset_name} from "
                f"{selected_path}: {error}"
            ) from error

        return pd.DataFrame(), selected_path

    return dataframe, selected_path


def ensure_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """
    Ensure specified columns exist without modifying the input object.
    """
    working = dataframe.copy()

    for column in columns:
        if column not in working.columns:
            working[column] = np.nan

    return working


def normalise_text(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series to lowercase searchable strings.
    """
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series to numeric values without raising errors.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_datetime(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series to datetime while tolerating mixed formats.
    """
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce",
        )
    except TypeError:
        # Compatibility with older pandas versions.
        return pd.to_datetime(
            series,
            errors="coerce",
        )


# -------------------------------------------------------------------
# Enriched-data preparation
# -------------------------------------------------------------------

def prepare_enriched_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the enriched unified dataset for dashboard use.
    """
    expected_columns = [
        "record_id",
        "record_type",
        "pillar",
        "indicator",
        "indicator_code",
        "value_numeric",
        "unit",
        "observation_date",
        "period_start",
        "fiscal_year",
        "category",
        "source_type",
        "source_name",
        "confidence",
        "gender",
        "location",
        "impact_direction",
        "impact_magnitude",
        "related_indicator",
        "parent_id",
        "lag_months",
    ]

    working = ensure_columns(
        dataframe,
        expected_columns,
    )

    working["value_numeric"] = safe_numeric(
        working["value_numeric"]
    )

    working["impact_magnitude"] = safe_numeric(
        working["impact_magnitude"]
    )

    working["lag_months"] = safe_numeric(
        working["lag_months"]
    )

    working["observation_date"] = safe_datetime(
        working["observation_date"]
    )

    working["period_start"] = safe_datetime(
        working["period_start"]
    )

    observation_year = (
        working["observation_date"]
        .dt.year
        .astype("Int64")
    )

    period_year = (
        working["period_start"]
        .dt.year
        .astype("Int64")
    )

    fiscal_year = (
        working["fiscal_year"]
        .astype("string")
        .str.extract(
            r"(\d{4})",
            expand=False,
        )
    )

    fiscal_year = pd.to_numeric(
        fiscal_year,
        errors="coerce",
    ).astype("Int64")

    working["analysis_year"] = (
        observation_year
        .fillna(period_year)
        .fillna(fiscal_year)
    )

    working["record_type_clean"] = normalise_text(
        working["record_type"]
    )

    working["indicator_code_clean"] = normalise_text(
        working["indicator_code"]
    )

    working["indicator_clean"] = normalise_text(
        working["indicator"]
    )

    working["pillar_clean"] = normalise_text(
        working["pillar"]
    )

    return working


def load_enriched_data() -> tuple[pd.DataFrame, Path]:
    """
    Load and prepare the primary enriched dataset.
    """
    dataframe, selected_path = safe_read_csv(
        ENRICHED_DATA_CANDIDATES,
        required=True,
        dataset_name="enriched financial inclusion dataset",
    )

    prepared = prepare_enriched_data(dataframe)

    return prepared, selected_path


# -------------------------------------------------------------------
# Dashboard-specific data extraction
# -------------------------------------------------------------------

def get_observations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return usable observation records.
    """
    if dataframe.empty:
        return pd.DataFrame()

    observations = dataframe[
        dataframe["record_type_clean"].eq("observation")
    ].copy()

    observations = observations.dropna(
        subset=[
            "analysis_year",
            "value_numeric",
        ]
    )

    observations["analysis_year"] = (
        observations["analysis_year"]
        .astype(int)
    )

    return observations


def get_events(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return event records from the unified dataset.
    """
    if dataframe.empty:
        return pd.DataFrame()

    events = dataframe[
        dataframe["record_type_clean"].eq("event")
    ].copy()

    return events.sort_values(
        "period_start",
        na_position="last",
    )


def get_targets(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return official target records.
    """
    if dataframe.empty:
        return pd.DataFrame()

    targets = dataframe[
        dataframe["record_type_clean"].eq("target")
    ].copy()

    return targets


def select_indicator(
    observations: pd.DataFrame,
    indicator_codes: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Select an indicator using exact codes first and keywords second.

    Results are aggregated by year to avoid duplicate plotting points.
    """
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "analysis_year",
                "value_numeric",
                "indicator",
                "indicator_code",
                "unit",
            ]
        )

    indicator_codes = indicator_codes or []
    keywords = keywords or []

    requested_codes = {
        str(code).strip().lower()
        for code in indicator_codes
    }

    code_mask = observations[
        "indicator_code_clean"
    ].isin(requested_codes)

    selected = observations[code_mask].copy()

    if selected.empty and keywords:
        keyword_mask = pd.Series(
            False,
            index=observations.index,
        )

        for keyword in keywords:
            keyword_mask = (
                keyword_mask
                | observations[
                    "indicator_clean"
                ].str.contains(
                    str(keyword).lower(),
                    regex=False,
                    na=False,
                )
            )

        selected = observations[
            keyword_mask
        ].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "analysis_year",
                "value_numeric",
                "indicator",
                "indicator_code",
                "unit",
            ]
        )

    # Prefer national or overall observations when possible.
    gender_text = normalise_text(
        selected["gender"]
    )

    location_text = normalise_text(
        selected["location"]
    )

    overall_gender_mask = gender_text.isin(
        [
            "",
            "all",
            "all adults",
            "total",
            "both",
            "national",
        ]
    )

    national_location_mask = location_text.isin(
        [
            "",
            "ethiopia",
            "national",
            "all",
            "total",
        ]
    )

    preferred = selected[
        overall_gender_mask
        & national_location_mask
    ].copy()

    if not preferred.empty:
        selected = preferred

    metadata = (
        selected.sort_values("analysis_year")
        .groupby("analysis_year", as_index=False)
        .agg(
            value_numeric=(
                "value_numeric",
                "mean",
            ),
            indicator=(
                "indicator",
                "first",
            ),
            indicator_code=(
                "indicator_code",
                "first",
            ),
            unit=(
                "unit",
                "first",
            ),
        )
    )

    return metadata.sort_values(
        "analysis_year"
    )


def get_account_ownership_series(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the national Account Ownership series.
    """
    return select_indicator(
        observations,
        indicator_codes=[
            "ACC_OWNERSHIP",
        ],
        keywords=[
            "account ownership",
        ],
    )


def get_mobile_money_series(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract mobile-money account penetration.
    """
    return select_indicator(
        observations,
        indicator_codes=[
            "ACC_MM_ACCOUNT",
        ],
        keywords=[
            "mobile money account",
            "mobile money ownership",
        ],
    )


def get_digital_payment_series(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract digital-payment usage observations.
    """
    return select_indicator(
        observations,
        indicator_codes=[
            "USG_DIGITAL_PAYMENT",
            "USG_DIGITAL_PAYMENTS",
            "USG_MADE_RECEIVED_DIGITAL_PAYMENT",
            "DIGITAL_PAYMENT_USAGE",
        ],
        keywords=[
            "made or received digital payment",
            "digital payment usage",
            "digital payment adoption",
        ],
    )


def calculate_growth_table(
    indicator_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate absolute and relative changes between observation years.
    """
    if indicator_data.empty:
        return pd.DataFrame(
            columns=[
                "analysis_year",
                "value_numeric",
                "previous_value",
                "change_pp",
                "relative_growth_percent",
                "years_elapsed",
                "annualised_change_pp",
            ]
        )

    growth = indicator_data[
        [
            "analysis_year",
            "value_numeric",
        ]
    ].copy()

    growth = growth.sort_values(
        "analysis_year"
    )

    growth["previous_value"] = (
        growth["value_numeric"].shift(1)
    )

    growth["previous_year"] = (
        growth["analysis_year"].shift(1)
    )

    growth["change_pp"] = (
        growth["value_numeric"]
        - growth["previous_value"]
    )

    growth["relative_growth_percent"] = (
        growth["change_pp"]
        / growth["previous_value"]
        * 100
    )

    growth["years_elapsed"] = (
        growth["analysis_year"]
        - growth["previous_year"]
    )

    growth["annualised_change_pp"] = (
        growth["change_pp"]
        / growth["years_elapsed"]
    )

    return growth


# -------------------------------------------------------------------
# Optional Task 3 and Task 4 outputs
# -------------------------------------------------------------------

def load_forecast_scenarios() -> tuple[pd.DataFrame, Path | None]:
    """
    Load Task 4 scenario forecasts when available.
    """
    dataframe, selected_path = safe_read_csv(
        FORECAST_SCENARIO_CANDIDATES,
        required=False,
        dataset_name="Task 4 scenario forecasts",
    )

    if dataframe.empty:
        return dataframe, selected_path

    dataframe = ensure_columns(
        dataframe,
        [
            "target",
            "year",
            "baseline_forecast",
            "lower_95_percent",
            "upper_95_percent",
            "event_adjustment_pp",
            "base_scenario",
            "optimistic_scenario",
            "pessimistic_scenario",
            "model_selected",
        ],
    )

    numeric_columns = [
        "year",
        "baseline_forecast",
        "lower_95_percent",
        "upper_95_percent",
        "event_adjustment_pp",
        "base_scenario",
        "optimistic_scenario",
        "pessimistic_scenario",
    ]

    for column in numeric_columns:
        dataframe[column] = safe_numeric(
            dataframe[column]
        )

    return dataframe, selected_path


def load_baseline_forecasts() -> tuple[pd.DataFrame, Path | None]:
    """
    Load Task 4 baseline forecasts when available.
    """
    dataframe, selected_path = safe_read_csv(
        BASELINE_FORECAST_CANDIDATES,
        required=False,
        dataset_name="Task 4 baseline forecasts",
    )

    return dataframe, selected_path


def load_event_summary() -> tuple[pd.DataFrame, Path | None]:
    """
    Load Task 3 event-impact summary when available.
    """
    dataframe, selected_path = safe_read_csv(
        EVENT_SUMMARY_CANDIDATES,
        required=False,
        dataset_name="Task 3 event-impact summary",
    )

    return dataframe, selected_path


def load_event_matrix() -> tuple[pd.DataFrame, Path | None]:
    """
    Load Task 3 event-indicator matrix when available.
    """
    dataframe, selected_path = safe_read_csv(
        EVENT_MATRIX_CANDIDATES,
        required=False,
        dataset_name="Task 3 event-indicator matrix",
    )

    return dataframe, selected_path


def load_target_progress() -> tuple[pd.DataFrame, Path | None]:
    """
    Load Task 4 target-progress output when available.
    """
    dataframe, selected_path = safe_read_csv(
        TARGET_PROGRESS_CANDIDATES,
        required=False,
        dataset_name="Task 4 target progress",
    )

    return dataframe, selected_path


def load_largest_event_impacts() -> tuple[pd.DataFrame, Path | None]:
    """
    Load the ranked Task 4 event-impact output when available.
    """
    dataframe, selected_path = safe_read_csv(
        LARGEST_EVENT_IMPACTS_CANDIDATES,
        required=False,
        dataset_name="Task 4 largest event impacts",
    )

    return dataframe, selected_path


# -------------------------------------------------------------------
# Summary metrics
# -------------------------------------------------------------------

def latest_indicator_value(
    indicator_data: pd.DataFrame,
) -> tuple[float | None, int | None]:
    """
    Return the most recent numeric value and year.
    """
    if indicator_data.empty:
        return None, None

    usable = indicator_data.dropna(
        subset=[
            "analysis_year",
            "value_numeric",
        ]
    )

    if usable.empty:
        return None, None

    latest = usable.sort_values(
        "analysis_year"
    ).iloc[-1]

    return (
        float(latest["value_numeric"]),
        int(latest["analysis_year"]),
    )


def previous_indicator_value(
    indicator_data: pd.DataFrame,
) -> tuple[float | None, int | None]:
    """
    Return the second-most-recent value and year.
    """
    if len(indicator_data) < 2:
        return None, None

    usable = indicator_data.dropna(
        subset=[
            "analysis_year",
            "value_numeric",
        ]
    ).sort_values("analysis_year")

    if len(usable) < 2:
        return None, None

    previous = usable.iloc[-2]

    return (
        float(previous["value_numeric"]),
        int(previous["analysis_year"]),
    )


def calculate_metric_delta(
    indicator_data: pd.DataFrame,
) -> float | None:
    """
    Calculate the percentage-point change between the latest
    two observations.
    """
    latest_value, _ = latest_indicator_value(
        indicator_data
    )

    previous_value, _ = previous_indicator_value(
        indicator_data
    )

    if latest_value is None or previous_value is None:
        return None

    return latest_value - previous_value