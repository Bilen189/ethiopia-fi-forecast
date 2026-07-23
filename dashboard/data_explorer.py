"""
Data Explorer and final dashboard utilities for the Ethiopia
Financial Inclusion Forecasting Dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def safe_text(series: pd.Series) -> pd.Series:
    """
    Convert a Series into safe displayable text.
    """
    return (
        series
        .fillna("Not specified")
        .astype(str)
        .str.strip()
    )


def render_data_explorer_page(
    financial_data: pd.DataFrame,
) -> None:
    """
    Render a filterable data explorer and download interface.
    """
    st.header("Data Explorer")

    st.markdown(
        """
        Filter and inspect the unified financial-inclusion dataset.
        The explorer includes observations, events, impact links and
        policy targets collected during Tasks 1–4.
        """
    )

    if financial_data.empty:
        st.warning(
            "The financial inclusion dataset is empty."
        )
        return

    explorer_data = financial_data.copy()

    filter_1, filter_2, filter_3 = st.columns(3)

    record_type_options = sorted(
        safe_text(
            explorer_data["record_type"]
        ).unique().tolist()
    )

    with filter_1:
        selected_record_types = st.multiselect(
            "Record type",
            options=record_type_options,
            default=record_type_options,
        )

    pillar_options = sorted(
        safe_text(
            explorer_data["pillar"]
        ).unique().tolist()
    )

    with filter_2:
        selected_pillars = st.multiselect(
            "Pillar",
            options=pillar_options,
            default=pillar_options,
        )

    confidence_options = sorted(
        safe_text(
            explorer_data["confidence"]
        ).unique().tolist()
    )

    with filter_3:
        selected_confidence = st.multiselect(
            "Confidence",
            options=confidence_options,
            default=confidence_options,
        )

    record_type_text = safe_text(
        explorer_data["record_type"]
    )

    pillar_text = safe_text(
        explorer_data["pillar"]
    )

    confidence_text = safe_text(
        explorer_data["confidence"]
    )

    filtered_data = explorer_data[
        record_type_text.isin(
            selected_record_types
        )
        & pillar_text.isin(
            selected_pillars
        )
        & confidence_text.isin(
            selected_confidence
        )
    ].copy()

    search_text = st.text_input(
        "Search indicator, source, event or notes",
        placeholder=(
            "Example: mobile money, Telebirr, policy, World Bank"
        ),
    )

    if search_text.strip():
        searchable_columns = [
            column
            for column in [
                "indicator",
                "indicator_code",
                "source_name",
                "category",
                "original_text",
                "notes",
                "record_id",
            ]
            if column in filtered_data.columns
        ]

        search_mask = pd.Series(
            False,
            index=filtered_data.index,
        )

        for column in searchable_columns:
            search_mask = (
                search_mask
                | filtered_data[column]
                .fillna("")
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    regex=False,
                    na=False,
                )
            )

        filtered_data = filtered_data[
            search_mask
        ]

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Filtered Records",
        f"{len(filtered_data):,}",
    )

    metric_2.metric(
        "Observations",
        f"{(
            filtered_data['record_type']
            .fillna('')
            .astype(str)
            .str.lower()
            .eq('observation')
            .sum()
        ):,}",
    )

    metric_3.metric(
        "Events",
        f"{(
            filtered_data['record_type']
            .fillna('')
            .astype(str)
            .str.lower()
            .eq('event')
            .sum()
        ):,}",
    )

    metric_4.metric(
        "Unique Indicators",
        f"{filtered_data['indicator_code'].nunique(dropna=True):,}",
    )

    st.divider()

    st.subheader("Filtered Dataset")

    preferred_columns = [
        "record_id",
        "record_type",
        "pillar",
        "indicator",
        "indicator_code",
        "value_numeric",
        "unit",
        "observation_date",
        "period_start",
        "category",
        "source_name",
        "confidence",
        "parent_id",
        "related_indicator",
        "impact_direction",
        "impact_magnitude",
        "lag_months",
        "notes",
    ]

    display_columns = [
        column
        for column in preferred_columns
        if column in filtered_data.columns
    ]

    st.dataframe(
        filtered_data[display_columns],
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    csv_data = filtered_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download filtered dataset",
        data=csv_data,
        file_name=(
            "ethiopia_financial_inclusion_filtered_data.csv"
        ),
        mime="text/csv",
    )

    st.divider()

    st.subheader("Record Distribution")

    distribution = (
        filtered_data["record_type"]
        .fillna("Not specified")
        .value_counts()
        .rename_axis("Record Type")
        .reset_index(name="Records")
    )

    if not distribution.empty:
        distribution_chart = px.bar(
            distribution,
            x="Record Type",
            y="Records",
            text="Records",
        )

        distribution_chart.update_traces(
            textposition="outside"
        )

        distribution_chart.update_layout(
            height=400,
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            distribution_chart,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Data Quality by Confidence")

    confidence_distribution = (
        filtered_data["confidence"]
        .fillna("Not specified")
        .value_counts()
        .rename_axis("Confidence")
        .reset_index(name="Records")
    )

    if not confidence_distribution.empty:
        confidence_chart = px.pie(
            confidence_distribution,
            names="Confidence",
            values="Records",
            hole=0.45,
        )

        confidence_chart.update_layout(
            height=420,
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
        )

        st.plotly_chart(
            confidence_chart,
            use_container_width=True,
        )

    with st.expander(
        "Dataset interpretation guidance"
    ):
        st.markdown(
            """
            - Observation records contain measured financial
              inclusion or infrastructure values.
            - Event records identify relevant policies, launches,
              investments and market milestones.
            - Impact-link records connect an event to a particular
              indicator and contain modeled effect assumptions.
            - Target records contain official policy objectives.
            - Confidence describes the reliability of a source or
              modeled estimate, not statistical confidence.
            """
        )