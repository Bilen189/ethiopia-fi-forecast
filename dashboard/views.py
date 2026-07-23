"""
Interactive analytical views for the Ethiopia Financial Inclusion
Forecasting Dashboard.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import (
    get_account_ownership_series,
    get_digital_payment_series,
    get_events,
    get_mobile_money_series,
    get_observations,
    load_baseline_forecasts,
    load_event_matrix,
    load_event_summary,
    load_forecast_scenarios,
    load_largest_event_impacts,
    load_target_progress,
)


# -------------------------------------------------------------------
# Generic utilities
# -------------------------------------------------------------------

def normalise_column_name(column: object) -> str:
    """
    Convert a column name to a lowercase underscore format.
    """
    return (
        str(column)
        .strip()
        .lower()
        .replace("%", "percent")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def normalise_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a copy with consistently formatted column names.
    """
    working = dataframe.copy()

    working.columns = [
        normalise_column_name(column)
        for column in working.columns
    ]

    return working


def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """
    Find the first matching column from a candidate list.
    """
    if dataframe.empty:
        return None

    normalised_mapping = {
        normalise_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalised_candidate = normalise_column_name(
            candidate
        )

        if normalised_candidate in normalised_mapping:
            return normalised_mapping[normalised_candidate]

    return None


def find_column_containing(
    dataframe: pd.DataFrame,
    keywords: Iterable[str],
) -> str | None:
    """
    Find a column containing all supplied keywords.
    """
    for column in dataframe.columns:
        column_text = normalise_column_name(column)

        if all(
            keyword.lower() in column_text
            for keyword in keywords
        ):
            return column

    return None


def safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series to numeric values.
    """
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_datetime(
    series: pd.Series,
) -> pd.Series:
    """
    Convert values to datetime safely.
    """
    try:
        return pd.to_datetime(
            series,
            format="mixed",
            errors="coerce",
        )
    except TypeError:
        return pd.to_datetime(
            series,
            errors="coerce",
        )


def safe_percent(
    value: float | int | None,
    digits: int = 1,
) -> str:
    """
    Format a percentage safely.
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}%"


def show_missing_data(
    title: str,
    checked_path: object | None = None,
) -> None:
    """
    Display a consistent missing-data message.
    """
    st.warning(
        f"{title} is unavailable because the required "
        "output file or columns were not found."
    )

    if checked_path is not None:
        st.caption(
            f"Last checked source: `{checked_path}`"
        )


def download_csv_button(
    dataframe: pd.DataFrame,
    label: str,
    filename: str,
    key: str,
) -> None:
    """
    Render a CSV download button.
    """
    if dataframe.empty:
        return

    csv_data = dataframe.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def target_display_name(
    value: object,
) -> str:
    """
    Create a readable target name.
    """
    text = str(value).strip()

    replacements = {
        "ACC_OWNERSHIP": "Account Ownership",
        "USG_DIGITAL_PAYMENT": "Digital Payment Usage",
        "USG_DIGITAL_PAYMENTS": "Digital Payment Usage",
        "Account_Ownership": "Account Ownership",
        "Digital_Payment_Usage": "Digital Payment Usage",
    }

    if text in replacements:
        return replacements[text]

    return (
        text
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# -------------------------------------------------------------------
# Trends page
# -------------------------------------------------------------------

def render_trends_page(
    financial_data: pd.DataFrame,
) -> None:
    """
    Render interactive time-series and channel-comparison views.
    """
    st.header("Trends")

    st.markdown(
        """
        Explore how Access, Usage and related digital-finance
        indicators have evolved over time. The controls can be used
        to limit the analysis period and compare multiple channels.
        """
    )

    observations = get_observations(
        financial_data
    )

    if observations.empty:
        show_missing_data(
            "Observation trends"
        )
        return

    available_years = sorted(
        observations["analysis_year"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not available_years:
        show_missing_data(
            "Observation years"
        )
        return

    minimum_year = min(available_years)
    maximum_year = max(available_years)

    selected_years = st.slider(
        "Select analysis period",
        min_value=minimum_year,
        max_value=maximum_year,
        value=(
            minimum_year,
            maximum_year,
        ),
        step=1,
    )

    filtered_observations = observations[
        observations["analysis_year"].between(
            selected_years[0],
            selected_years[1],
        )
    ].copy()

    indicator_options = (
        filtered_observations[
            [
                "indicator_code",
                "indicator",
            ]
        ]
        .dropna(how="all")
        .drop_duplicates()
    )

    indicator_options["display_name"] = (
        indicator_options["indicator"]
        .fillna(
            indicator_options["indicator_code"]
        )
        .astype(str)
    )

    display_to_code = {
        row["display_name"]: row["indicator_code"]
        for _, row in indicator_options.iterrows()
    }

    preferred_names = []

    for preferred_keyword in [
        "account ownership",
        "mobile money",
        "digital payment",
        "p2p",
        "atm",
    ]:
        matches = [
            display_name
            for display_name in display_to_code
            if preferred_keyword
            in display_name.lower()
        ]

        preferred_names.extend(matches)

    preferred_names = list(
        dict.fromkeys(preferred_names)
    )[:5]

    if not preferred_names:
        preferred_names = list(
            display_to_code.keys()
        )[:3]

    selected_indicators = st.multiselect(
        "Select indicators",
        options=list(display_to_code.keys()),
        default=preferred_names,
    )

    if not selected_indicators:
        st.info(
            "Select at least one indicator to display the trend."
        )
        return

    selected_codes = {
        str(display_to_code[name]).strip()
        for name in selected_indicators
    }

    trend_data = filtered_observations[
        filtered_observations[
            "indicator_code"
        ].astype(str).isin(selected_codes)
    ].copy()

    if trend_data.empty:
        # Fall back to matching by displayed indicator names.
        trend_data = filtered_observations[
            filtered_observations[
                "indicator"
            ].astype(str).isin(selected_indicators)
        ].copy()

    trend_data["display_indicator"] = (
        trend_data["indicator"]
        .fillna(
            trend_data["indicator_code"]
        )
        .astype(str)
    )

    trend_summary = (
        trend_data
        .groupby(
            [
                "analysis_year",
                "display_indicator",
            ],
            as_index=False,
        )
        .agg(
            value_numeric=(
                "value_numeric",
                "mean",
            )
        )
    )

    st.subheader("Interactive Indicator Trends")

    trend_chart = px.line(
        trend_summary,
        x="analysis_year",
        y="value_numeric",
        color="display_indicator",
        markers=True,
        labels={
            "analysis_year": "Year",
            "value_numeric": "Indicator value",
            "display_indicator": "Indicator",
        },
    )

    trend_chart.update_traces(
        line={"width": 3},
        marker={"size": 8},
    )

    trend_chart.update_layout(
        height=500,
        hovermode="x unified",
        legend_title_text="Indicator",
        margin={
            "l": 20,
            "r": 20,
            "t": 20,
            "b": 20,
        },
    )

    st.plotly_chart(
        trend_chart,
        use_container_width=True,
    )

    st.caption(
        "Values may use different units. Interpret comparisons "
        "carefully and consult indicator definitions."
    )

    st.divider()

    st.subheader("Core Channel Comparison")

    account_series = get_account_ownership_series(
        observations
    )

    mobile_series = get_mobile_money_series(
        observations
    )

    digital_series = get_digital_payment_series(
        observations
    )

    channel_frames = []

    for name, dataframe in [
        (
            "Account Ownership",
            account_series,
        ),
        (
            "Mobile-Money Account",
            mobile_series,
        ),
        (
            "Digital Payment Usage",
            digital_series,
        ),
    ]:
        if dataframe.empty:
            continue

        frame = dataframe[
            [
                "analysis_year",
                "value_numeric",
            ]
        ].copy()

        frame = frame[
            frame["analysis_year"].between(
                selected_years[0],
                selected_years[1],
            )
        ]

        frame["Channel"] = name
        channel_frames.append(frame)

    if channel_frames:
        channel_data = pd.concat(
            channel_frames,
            ignore_index=True,
        )

        channel_chart = px.line(
            channel_data,
            x="analysis_year",
            y="value_numeric",
            color="Channel",
            markers=True,
            labels={
                "analysis_year": "Year",
                "value_numeric": "Share of adults (%)",
            },
        )

        channel_chart.update_traces(
            line={"width": 3},
            marker={"size": 9},
        )

        channel_chart.update_layout(
            height=460,
            hovermode="x unified",
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            channel_chart,
            use_container_width=True,
        )
    else:
        show_missing_data(
            "Core channel comparison"
        )

    st.divider()

    st.subheader("Indicator Coverage by Year")

    coverage_data = (
        filtered_observations
        .groupby(
            [
                "indicator_code",
                "analysis_year",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "record_count",
            }
        )
    )

    if coverage_data.empty:
        show_missing_data(
            "Indicator coverage"
        )
    else:
        coverage_chart = px.scatter(
            coverage_data,
            x="analysis_year",
            y="indicator_code",
            size="record_count",
            labels={
                "analysis_year": "Year",
                "indicator_code": "Indicator",
                "record_count": "Records",
            },
        )

        coverage_chart.update_layout(
            height=max(
                450,
                len(
                    coverage_data[
                        "indicator_code"
                    ].unique()
                ) * 22,
            ),
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            coverage_chart,
            use_container_width=True,
        )

    download_csv_button(
        trend_summary,
        "Download selected trend data",
        "selected_financial_inclusion_trends.csv",
        "download_trends",
    )


# -------------------------------------------------------------------
# Event-impact page
# -------------------------------------------------------------------

def prepare_event_matrix(
    matrix_data: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """
    Prepare an event-indicator association matrix for plotting.
    """
    if matrix_data.empty:
        return pd.DataFrame(), None

    matrix = matrix_data.copy()

    unnamed_columns = [
        column
        for column in matrix.columns
        if str(column).lower().startswith(
            "unnamed"
        )
    ]

    if unnamed_columns:
        matrix = matrix.drop(
            columns=unnamed_columns
        )

    event_column = find_column(
        matrix,
        [
            "event",
            "event_name",
            "event_title",
            "description",
            "record_id",
        ],
    )

    if event_column is None:
        first_column = matrix.columns[0]

        if not pd.api.types.is_numeric_dtype(
            matrix[first_column]
        ):
            event_column = first_column

    numeric_columns = []

    for column in matrix.columns:
        if column == event_column:
            continue

        converted = safe_numeric(
            matrix[column]
        )

        if converted.notna().any():
            matrix[column] = converted
            numeric_columns.append(column)

    if not numeric_columns:
        return pd.DataFrame(), event_column

    if event_column is None:
        matrix["Event"] = [
            f"Event {index + 1}"
            for index in range(len(matrix))
        ]
        event_column = "Event"

    prepared = matrix[
        [event_column] + numeric_columns
    ].copy()

    prepared[event_column] = (
        prepared[event_column]
        .fillna("Unknown event")
        .astype(str)
    )

    return prepared, event_column


def render_event_impacts_page(
    financial_data: pd.DataFrame,
) -> None:
    """
    Render event timelines and event-indicator effects.
    """
    st.header("Event Impacts")

    st.markdown(
        """
        This page presents the modeled relationships between
        policy changes, product launches, infrastructure milestones
        and financial inclusion indicators. These are association
        estimates rather than experimentally identified causal effects.
        """
    )

    events = get_events(
        financial_data
    )

    st.subheader("Financial-Inclusion Event Timeline")

    if events.empty:
        show_missing_data(
            "Event timeline"
        )
    else:
        timeline = events.copy()

        date_column = None

        for candidate in [
            "period_start",
            "observation_date",
        ]:
            if (
                candidate in timeline.columns
                and timeline[candidate].notna().any()
            ):
                date_column = candidate
                break

        label_column = find_column(
            timeline,
            [
                "event_name",
                "title",
                "indicator",
                "description",
                "original_text",
                "record_id",
            ],
        )

        if label_column is None:
            timeline["event_label"] = (
                "Financial inclusion event"
            )
            label_column = "event_label"

        category_column = find_column(
            timeline,
            [
                "category",
                "event_type",
                "source_type",
            ],
        )

        if category_column is None:
            timeline["event_category"] = (
                "Event"
            )
            category_column = "event_category"

        if date_column is None:
            show_missing_data(
                "Dated event timeline"
            )
        else:
            timeline[date_column] = safe_datetime(
                timeline[date_column]
            )

            timeline = timeline.dropna(
                subset=[date_column]
            )

            timeline["Timeline level"] = (
                timeline[category_column]
                .fillna("Event")
                .astype(str)
            )

            event_chart = px.scatter(
                timeline,
                x=date_column,
                y="Timeline level",
                color="Timeline level",
                hover_name=label_column,
                labels={
                    date_column: "Event date",
                    "Timeline level": "Event category",
                },
            )

            event_chart.update_traces(
                marker={
                    "size": 15,
                    "line": {
                        "width": 1,
                    },
                }
            )

            event_chart.update_layout(
                height=440,
                showlegend=False,
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 20,
                    "b": 20,
                },
            )

            st.plotly_chart(
                event_chart,
                use_container_width=True,
            )

    st.divider()

    matrix_data, matrix_path = (
        load_event_matrix()
    )

    if matrix_data.empty:
        summary_data, summary_path = (
            load_event_summary()
        )

        if not summary_data.empty:
            normalised = normalise_columns(
                summary_data
            )

            event_column = find_column(
                normalised,
                [
                    "event",
                    "event_name",
                    "event_title",
                    "parent_id",
                ],
            )

            indicator_column = find_column(
                normalised,
                [
                    "indicator",
                    "related_indicator",
                    "indicator_code",
                ],
            )

            effect_column = find_column(
                normalised,
                [
                    "estimated_effect",
                    "effect",
                    "impact_magnitude",
                    "signed_impact",
                    "weighted_impact",
                ],
            )

            if (
                event_column
                and indicator_column
                and effect_column
            ):
                normalised[effect_column] = safe_numeric(
                    normalised[effect_column]
                )

                matrix_data = normalised.pivot_table(
                    index=event_column,
                    columns=indicator_column,
                    values=effect_column,
                    aggfunc="sum",
                    fill_value=0,
                ).reset_index()

                matrix_path = summary_path

    st.subheader(
        "Event–Indicator Association Matrix"
    )

    prepared_matrix, event_column = (
        prepare_event_matrix(
            matrix_data
        )
    )

    if prepared_matrix.empty or event_column is None:
        show_missing_data(
            "Event–indicator association matrix",
            matrix_path,
        )
    else:
        matrix_indexed = prepared_matrix.set_index(
            event_column
        )

        matrix_chart = px.imshow(
            matrix_indexed,
            text_auto=".2f",
            aspect="auto",
            labels={
                "x": "Financial inclusion indicator",
                "y": "Event",
                "color": "Estimated effect",
            },
        )

        matrix_chart.update_layout(
            height=max(
                450,
                len(matrix_indexed) * 38,
            ),
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            matrix_chart,
            use_container_width=True,
        )

        st.caption(
            "Positive values indicate an expected increase in "
            "the associated indicator. Negative values indicate "
            "a possible reduction."
        )

        download_csv_button(
            prepared_matrix,
            "Download association matrix",
            "event_indicator_association_matrix.csv",
            "download_event_matrix",
        )

    st.divider()

    largest_impacts, impact_path = (
        load_largest_event_impacts()
    )

    st.subheader("Largest Modeled Event Effects")

    if largest_impacts.empty:
        show_missing_data(
            "Ranked event impacts",
            impact_path,
        )
    else:
        ranked = normalise_columns(
            largest_impacts
        )

        event_column = find_column(
            ranked,
            [
                "event_name",
                "event",
                "event_title",
                "parent_id",
            ],
        )

        target_column = find_column(
            ranked,
            [
                "target",
                "indicator",
                "related_indicator",
            ],
        )

        impact_column = find_column(
            ranked,
            [
                "event_adjustment_pp",
                "weighted_effect",
                "estimated_effect",
                "impact_magnitude",
                "absolute_impact",
                "signed_impact",
            ],
        )

        if impact_column is None:
            impact_column = find_column_containing(
                ranked,
                ["impact"],
            )

        if event_column and impact_column:
            ranked[impact_column] = safe_numeric(
                ranked[impact_column]
            )

            ranked = ranked.dropna(
                subset=[impact_column]
            )

            ranked["absolute_effect"] = (
                ranked[impact_column].abs()
            )

            ranked = ranked.sort_values(
                "absolute_effect",
                ascending=False,
            ).head(15)

            if target_column:
                ranked["display_target"] = (
                    ranked[target_column]
                    .map(target_display_name)
                )
            else:
                ranked["display_target"] = (
                    "Financial Inclusion"
                )

            impact_chart = px.bar(
                ranked,
                x=impact_column,
                y=event_column,
                color="display_target",
                orientation="h",
                labels={
                    impact_column: (
                        "Estimated effect "
                        "(percentage points)"
                    ),
                    event_column: "Event",
                    "display_target": "Target",
                },
            )

            impact_chart.update_layout(
                height=max(
                    450,
                    len(ranked) * 35,
                ),
                yaxis={
                    "categoryorder": "total ascending",
                },
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 20,
                    "b": 20,
                },
            )

            st.plotly_chart(
                impact_chart,
                use_container_width=True,
            )
        else:
            st.dataframe(
                largest_impacts,
                use_container_width=True,
                hide_index=True,
            )

        download_csv_button(
            largest_impacts,
            "Download ranked impacts",
            "largest_event_impacts.csv",
            "download_largest_impacts",
        )

    with st.expander(
        "How to interpret the impact model"
    ):
        st.markdown(
            """
            The event model translates each impact-link record
            into an estimated percentage-point effect. The model
            incorporates the direction, magnitude and lag attached
            to the event–indicator relationship.

            Multiple event effects may overlap. Results therefore
            describe modeled associations under explicit assumptions;
            they should not be treated as proof that an event alone
            caused the observed change.
            """
        )


# -------------------------------------------------------------------
# Forecast preparation
# -------------------------------------------------------------------

def prepare_forecast_data(
    forecast_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize Task 4 forecast outputs.
    """
    if forecast_data.empty:
        return pd.DataFrame()

    forecast = normalise_columns(
        forecast_data
    )

    rename_map = {}

    target_column = find_column(
        forecast,
        [
            "target",
            "indicator",
            "target_name",
            "series",
        ],
    )

    year_column = find_column(
        forecast,
        [
            "year",
            "forecast_year",
            "analysis_year",
        ],
    )

    baseline_column = find_column(
        forecast,
        [
            "baseline_forecast",
            "forecast",
            "predicted_value",
            "point_forecast",
            "linear_forecast",
        ],
    )

    lower_column = find_column(
        forecast,
        [
            "lower_95_percent",
            "lower_95",
            "lower_ci",
            "prediction_lower",
            "lower_bound",
        ],
    )

    upper_column = find_column(
        forecast,
        [
            "upper_95_percent",
            "upper_95",
            "upper_ci",
            "prediction_upper",
            "upper_bound",
        ],
    )

    base_column = find_column(
        forecast,
        [
            "base_scenario",
            "base",
            "event_adjusted_forecast",
        ],
    )

    optimistic_column = find_column(
        forecast,
        [
            "optimistic_scenario",
            "optimistic",
        ],
    )

    pessimistic_column = find_column(
        forecast,
        [
            "pessimistic_scenario",
            "pessimistic",
        ],
    )

    model_column = find_column(
        forecast,
        [
            "model_selected",
            "model",
            "model_name",
        ],
    )

    mappings = {
        target_column: "target",
        year_column: "year",
        baseline_column: "baseline_forecast",
        lower_column: "lower_95",
        upper_column: "upper_95",
        base_column: "base_scenario",
        optimistic_column: "optimistic_scenario",
        pessimistic_column: "pessimistic_scenario",
        model_column: "model",
    }

    for original, replacement in mappings.items():
        if original is not None:
            rename_map[original] = replacement

    forecast = forecast.rename(
        columns=rename_map
    )

    required_columns = [
        "target",
        "year",
        "baseline_forecast",
        "lower_95",
        "upper_95",
        "base_scenario",
        "optimistic_scenario",
        "pessimistic_scenario",
        "model",
    ]

    for column in required_columns:
        if column not in forecast.columns:
            forecast[column] = np.nan

    if forecast["target"].isna().all():
        forecast["target"] = (
            "Financial Inclusion"
        )

    numeric_columns = [
        "year",
        "baseline_forecast",
        "lower_95",
        "upper_95",
        "base_scenario",
        "optimistic_scenario",
        "pessimistic_scenario",
    ]

    for column in numeric_columns:
        forecast[column] = safe_numeric(
            forecast[column]
        )

    forecast = forecast.dropna(
        subset=[
            "year",
            "baseline_forecast",
        ]
    )

    forecast["year"] = (
        forecast["year"].astype(int)
    )

    forecast["display_target"] = (
        forecast["target"]
        .map(target_display_name)
    )

    return forecast.sort_values(
        [
            "display_target",
            "year",
        ]
    )


def render_confidence_forecast_chart(
    forecast: pd.DataFrame,
    forecast_value_column: str,
    chart_title: str,
) -> None:
    """
    Plot a point forecast and confidence interval.
    """
    figure = go.Figure()

    for target in forecast[
        "display_target"
    ].dropna().unique():
        target_data = forecast[
            forecast["display_target"].eq(
                target
            )
        ].sort_values("year")

        if (
            target_data["lower_95"].notna().any()
            and target_data["upper_95"].notna().any()
        ):
            figure.add_trace(
                go.Scatter(
                    x=pd.concat(
                        [
                            target_data["year"],
                            target_data[
                                "year"
                            ].iloc[::-1],
                        ]
                    ),
                    y=pd.concat(
                        [
                            target_data[
                                "upper_95"
                            ],
                            target_data[
                                "lower_95"
                            ].iloc[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=(
                        "rgba(100, 100, 100, 0.15)"
                    ),
                    line={
                        "color": "rgba(0,0,0,0)",
                    },
                    hoverinfo="skip",
                    name=f"{target} 95% interval",
                    showlegend=True,
                )
            )

        figure.add_trace(
            go.Scatter(
                x=target_data["year"],
                y=target_data[
                    forecast_value_column
                ],
                mode="lines+markers",
                name=target,
                line={"width": 3},
                marker={"size": 9},
                hovertemplate=(
                    f"<b>{target}</b><br>"
                    "Year: %{x}<br>"
                    "Forecast: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=chart_title,
        xaxis_title="Year",
        yaxis_title="Share of adults (%)",
        height=500,
        hovermode="x unified",
        margin={
            "l": 20,
            "r": 20,
            "t": 55,
            "b": 20,
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Forecasts page
# -------------------------------------------------------------------

def render_forecasts_page() -> None:
    """
    Render Task 4 forecasts with confidence intervals.
    """
    st.header("Forecasts")

    st.markdown(
        """
        Review the projected Access and Usage outcomes for
        2025–2027. Confidence intervals reflect the uncertainty
        created by the sparse historical Findex series.
        """
    )

    scenario_data, scenario_path = (
        load_forecast_scenarios()
    )

    baseline_data, baseline_path = (
        load_baseline_forecasts()
    )

    if not scenario_data.empty:
        raw_forecast = scenario_data
        loaded_path = scenario_path
    else:
        raw_forecast = baseline_data
        loaded_path = baseline_path

    forecast = prepare_forecast_data(
        raw_forecast
    )

    if forecast.empty:
        show_missing_data(
            "Task 4 forecasts",
            loaded_path,
        )

        st.code(
            "models/task4_forecast_scenarios_2025_2027.csv"
        )
        return

    target_options = sorted(
        forecast["display_target"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_targets = st.multiselect(
        "Forecast targets",
        options=target_options,
        default=target_options,
    )

    filtered_forecast = forecast[
        forecast["display_target"].isin(
            selected_targets
        )
    ].copy()

    available_models = (
        filtered_forecast["model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if available_models:
        selected_model = st.selectbox(
            "Model selection",
            options=["All models"] + available_models,
        )

        if selected_model != "All models":
            filtered_forecast = filtered_forecast[
                filtered_forecast[
                    "model"
                ].astype(str).eq(
                    selected_model
                )
            ]

    if filtered_forecast.empty:
        st.info(
            "No forecasts match the selected filters."
        )
        return

    render_confidence_forecast_chart(
        filtered_forecast,
        "baseline_forecast",
        "Baseline Forecast with Approximate 95% Prediction Intervals",
    )

    st.caption(
        "Prediction intervals are intentionally wide because "
        "the target series contains only a small number of "
        "survey observations."
    )

    st.divider()

    st.subheader("Projected Milestones")

    latest_year = int(
        filtered_forecast["year"].max()
    )

    milestone_data = filtered_forecast[
        filtered_forecast["year"].eq(
            latest_year
        )
    ]

    milestone_columns = st.columns(
        max(
            1,
            min(
                4,
                len(milestone_data),
            ),
        )
    )

    for index, (_, row) in enumerate(
        milestone_data.iterrows()
    ):
        with milestone_columns[
            index % len(milestone_columns)
        ]:
            st.metric(
                label=(
                    f"{row['display_target']} "
                    f"({latest_year})"
                ),
                value=safe_percent(
                    row["baseline_forecast"]
                ),
            )

    st.dataframe(
        filtered_forecast[
            [
                "display_target",
                "year",
                "baseline_forecast",
                "lower_95",
                "upper_95",
                "model",
            ]
        ].rename(
            columns={
                "display_target": "Target",
                "year": "Year",
                "baseline_forecast": (
                    "Baseline Forecast (%)"
                ),
                "lower_95": "Lower 95% (%)",
                "upper_95": "Upper 95% (%)",
                "model": "Selected Model",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    download_csv_button(
        filtered_forecast,
        "Download forecast table",
        "task4_dashboard_forecasts.csv",
        "download_forecasts",
    )

    with st.expander(
        "Forecast interpretation and limitations"
    ):
        st.markdown(
            """
            - The baseline extends the historical trend without
              assuming unusually strong or weak implementation.
            - Event-adjusted scenarios incorporate modeled effects
              from policies, products and infrastructure.
            - Forecasts are not precise predictions. They describe
              plausible trajectories given limited historical data.
            - Survey definitions, active-account behavior and the
              distinction between registrations and unique users
              remain important sources of uncertainty.
            """
        )


# -------------------------------------------------------------------
# Inclusion projections page
# -------------------------------------------------------------------

def render_inclusion_projections_page() -> None:
    """
    Render scenario-based projections and target progress.
    """
    st.header("Inclusion Projections")

    st.markdown(
        """
        Compare optimistic, base and pessimistic financial
        inclusion scenarios and assess progress toward the
        60% Account Ownership target.
        """
    )

    raw_forecast, forecast_path = (
        load_forecast_scenarios()
    )

    forecast = prepare_forecast_data(
        raw_forecast
    )

    if forecast.empty:
        show_missing_data(
            "Scenario projections",
            forecast_path,
        )
        return

    scenario_mapping = {
        "Baseline trend": "baseline_forecast",
        "Base scenario": "base_scenario",
        "Optimistic scenario": (
            "optimistic_scenario"
        ),
        "Pessimistic scenario": (
            "pessimistic_scenario"
        ),
    }

    available_scenarios = {
        label: column
        for label, column
        in scenario_mapping.items()
        if (
            column in forecast.columns
            and forecast[column].notna().any()
        )
    }

    selected_scenario_label = st.selectbox(
        "Select projection scenario",
        options=list(
            available_scenarios.keys()
        ),
        index=(
            list(
                available_scenarios.keys()
            ).index("Base scenario")
            if "Base scenario"
            in available_scenarios
            else 0
        ),
    )

    selected_scenario_column = (
        available_scenarios[
            selected_scenario_label
        ]
    )

    figure = go.Figure()

    for target in forecast[
        "display_target"
    ].unique():
        target_data = forecast[
            forecast["display_target"].eq(
                target
            )
        ].sort_values("year")

        figure.add_trace(
            go.Scatter(
                x=target_data["year"],
                y=target_data[
                    selected_scenario_column
                ],
                mode="lines+markers",
                name=target,
                line={"width": 4},
                marker={"size": 10},
                hovertemplate=(
                    f"<b>{target}</b><br>"
                    "Year: %{x}<br>"
                    "Projection: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_hline(
        y=60,
        line_dash="dash",
        annotation_text="60% policy target",
        annotation_position="top left",
    )

    figure.update_layout(
        title=selected_scenario_label,
        xaxis_title="Year",
        yaxis_title="Share of adults (%)",
        height=500,
        hovermode="x unified",
        margin={
            "l": 20,
            "r": 20,
            "t": 55,
            "b": 20,
        },
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.divider()

    account_mask = (
        forecast["display_target"]
        .str.lower()
        .str.contains(
            "account ownership",
            na=False,
        )
    )

    account_projection = forecast[
        account_mask
    ].copy()

    st.subheader(
        "Progress Toward the 60% Access Target"
    )

    if account_projection.empty:
        show_missing_data(
            "Account Ownership target projection"
        )
    else:
        account_projection[
            "selected_projection"
        ] = account_projection[
            selected_scenario_column
        ]

        account_projection["gap_to_target"] = (
            60
            - account_projection[
                "selected_projection"
            ]
        )

        account_projection[
            "target_progress_percent"
        ] = (
            account_projection[
                "selected_projection"
            ]
            / 60
            * 100
        ).clip(
            lower=0,
            upper=100,
        )

        latest_projection = (
            account_projection
            .sort_values("year")
            .iloc[-1]
        )

        metric_1, metric_2, metric_3 = (
            st.columns(3)
        )

        metric_1.metric(
            "Projected Account Ownership",
            safe_percent(
                latest_projection[
                    "selected_projection"
                ]
            ),
        )

        metric_2.metric(
            "Gap to 60% Target",
            (
                f"{latest_projection['gap_to_target']:.1f} pp"
            ),
        )

        target_status = (
            "Target reached"
            if latest_projection[
                "selected_projection"
            ] >= 60
            else "Target not yet reached"
        )

        metric_3.metric(
            "Target Status",
            target_status,
        )

        progress_chart = go.Figure()

        progress_chart.add_trace(
            go.Bar(
                x=account_projection["year"],
                y=account_projection[
                    "selected_projection"
                ],
                name="Projected ownership",
                text=account_projection[
                    "selected_projection"
                ].round(1),
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )
        )

        progress_chart.add_hline(
            y=60,
            line_dash="dash",
            annotation_text="60% target",
            annotation_position="top left",
        )

        progress_chart.update_layout(
            xaxis_title="Year",
            yaxis_title=(
                "Account Ownership (%)"
            ),
            height=440,
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            progress_chart,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "How Scenarios Differ"
    )

    scenario_columns = [
        column
        for column in [
            "pessimistic_scenario",
            "base_scenario",
            "optimistic_scenario",
        ]
        if forecast[column].notna().any()
    ]

    if scenario_columns:
        scenario_long = forecast.melt(
            id_vars=[
                "display_target",
                "year",
            ],
            value_vars=scenario_columns,
            var_name="scenario",
            value_name="projection",
        )

        scenario_long["scenario"] = (
            scenario_long["scenario"]
            .str.replace(
                "_scenario",
                "",
                regex=False,
            )
            .str.title()
        )

        scenario_chart = px.line(
            scenario_long,
            x="year",
            y="projection",
            color="scenario",
            facet_col="display_target",
            markers=True,
            labels={
                "year": "Year",
                "projection": (
                    "Projected share (%)"
                ),
                "scenario": "Scenario",
                "display_target": "Target",
            },
        )

        scenario_chart.update_traces(
            line={"width": 3},
            marker={"size": 8},
        )

        scenario_chart.for_each_annotation(
            lambda annotation: annotation.update(
                text=annotation.text.split(
                    "="
                )[-1]
            )
        )

        scenario_chart.update_layout(
            height=470,
            margin={
                "l": 20,
                "r": 20,
                "t": 40,
                "b": 20,
            },
        )

        st.plotly_chart(
            scenario_chart,
            use_container_width=True,
        )

    st.divider()

    st.subheader(
        "Answers to the Consortium's Questions"
    )

    st.markdown(
        """
        **What appears to drive inclusion?**  
        Account growth is associated with product availability,
        interoperability, infrastructure coverage, agent access,
        digital identification and policy implementation. Continued
        usage also depends on relevant payment use cases.

        **Why can registrations rise faster than inclusion?**  
        Operator registration totals may include inactive accounts,
        duplicate accounts and users who already held bank accounts.
        Registered mobile-money accounts therefore do not translate
        directly into additional unique financially included adults.

        **What could have the largest future effect?**  
        Successful implementation of interoperable payments,
        merchant acceptance, improved network access, digital ID and
        products that encourage repeated usage are likely to matter
        more than registrations alone.

        **What is most uncertain?**  
        The greatest uncertainties are sparse survey observations,
        differences between data sources, unknown active-user rates,
        implementation quality and the lag between an intervention
        and measurable Findex outcomes.
        """
    )

    download_csv_button(
        forecast,
        "Download scenario projections",
        "financial_inclusion_scenario_projections.csv",
        "download_scenarios",
    )