"""
Interactive dashboard for the Ethiopia Financial Inclusion
Forecasting project.

Run locally with:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------------------------
# Import setup
# -------------------------------------------------------------------

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from dashboard.data_loader import (  # noqa: E402
    calculate_growth_table,
    calculate_metric_delta,
    get_account_ownership_series,
    get_digital_payment_series,
    get_events,
    get_mobile_money_series,
    get_observations,
    get_targets,
    latest_indicator_value,
    load_enriched_data,
)


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Ethiopia Financial Inclusion Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 12px;
            padding: 16px;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.92rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 700;
        }

        .dashboard-note {
            padding: 1rem;
            border-radius: 10px;
            background-color: rgba(49, 130, 206, 0.08);
            border-left: 4px solid #3182CE;
            margin-bottom: 1rem;
        }

        .small-muted {
            color: #6B7280;
            font-size: 0.88rem;
        }

        .section-gap {
            margin-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Cached data loading
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_dashboard_data():
    """
    Load the enriched financial inclusion dataset.
    """
    return load_enriched_data()


try:
    financial_data, loaded_data_path = (
        load_dashboard_data()
    )
except Exception as error:
    st.error(
        "The dashboard could not load the enriched financial "
        "inclusion dataset."
    )

    st.exception(error)

    st.info(
        "Confirm that the following file exists:\n\n"
        "`data/processed/ethiopia_fi_enriched.csv`"
    )

    st.stop()


observations = get_observations(
    financial_data
)

events = get_events(
    financial_data
)

targets = get_targets(
    financial_data
)

account_ownership = get_account_ownership_series(
    observations
)

mobile_money = get_mobile_money_series(
    observations
)

digital_payments = get_digital_payment_series(
    observations
)

account_growth = calculate_growth_table(
    account_ownership
)


# -------------------------------------------------------------------
# Reusable formatting
# -------------------------------------------------------------------

def format_percent(
    value: float | None,
    digits: int = 1,
) -> str:
    """
    Format a percentage value safely.
    """
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}%"


def format_delta(
    value: float | None,
) -> str | None:
    """
    Format a percentage-point metric delta.
    """
    if value is None or pd.isna(value):
        return None

    return f"{value:+.1f} pp"


def empty_chart_message(
    description: str,
) -> None:
    """
    Show a consistent warning when an indicator is unavailable.
    """
    st.warning(
        f"{description} could not be displayed because the "
        "required indicator was not found in the enriched dataset."
    )


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------

with st.sidebar:
    st.title("Financial Inclusion")

    st.caption(
        "Ethiopia forecasting and event-impact dashboard"
    )

    selected_page = st.radio(
        "Navigate",
        options=[
            "Overview",
            "Trends",
            "Event Impacts",
            "Forecasts",
            "Inclusion Projections",
            "Data Explorer",
        ],
        index=0,
    )

    st.divider()

    st.subheader("Data status")

    st.success(
        f"{len(financial_data):,} unified records loaded"
    )

    st.caption(
        f"Source: `{loaded_data_path}`"
    )

    st.metric(
        "Observations",
        f"{len(observations):,}",
    )

    st.metric(
        "Events",
        f"{len(events):,}",
    )

    st.metric(
        "Policy targets",
        f"{len(targets):,}",
    )

    st.divider()

    st.caption(
        "Built for Selam Analytics and Ethiopia's "
        "financial-inclusion stakeholders."
    )


# -------------------------------------------------------------------
# Dashboard title
# -------------------------------------------------------------------

st.title(
    "Ethiopia Financial Inclusion Forecasting Dashboard"
)

st.caption(
    "Tracking Access, Usage, event impacts and progress "
    "toward national financial-inclusion targets."
)


# -------------------------------------------------------------------
# Overview page
# -------------------------------------------------------------------

if selected_page == "Overview":
    st.header("Overview")

    st.markdown(
        """
        <div class="dashboard-note">
            This page summarizes Ethiopia's financial inclusion
            trajectory using the enriched unified dataset. Access
            refers to account ownership, while Usage refers to
            adults making or receiving digital payments.
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_account, latest_account_year = (
        latest_indicator_value(
            account_ownership
        )
    )

    latest_mobile_money, mobile_money_year = (
        latest_indicator_value(
            mobile_money
        )
    )

    latest_digital_payment, digital_payment_year = (
        latest_indicator_value(
            digital_payments
        )
    )

    account_delta = calculate_metric_delta(
        account_ownership
    )

    mobile_money_delta = calculate_metric_delta(
        mobile_money
    )

    digital_payment_delta = calculate_metric_delta(
        digital_payments
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    with metric_1:
        st.metric(
            label=(
                "Account Ownership"
                if latest_account_year is None
                else
                f"Account Ownership ({latest_account_year})"
            ),
            value=format_percent(
                latest_account
            ),
            delta=format_delta(
                account_delta
            ),
            help=(
                "Share of adults with an account at a "
                "financial institution or mobile-money service."
            ),
        )

    with metric_2:
        st.metric(
            label=(
                "Mobile-Money Account"
                if mobile_money_year is None
                else
                f"Mobile-Money Account ({mobile_money_year})"
            ),
            value=format_percent(
                latest_mobile_money
            ),
            delta=format_delta(
                mobile_money_delta
            ),
            help=(
                "Share of adults reporting ownership or use "
                "of a mobile-money account."
            ),
        )

    with metric_3:
        st.metric(
            label=(
                "Digital Payment Usage"
                if digital_payment_year is None
                else
                f"Digital Payment Usage ({digital_payment_year})"
            ),
            value=format_percent(
                latest_digital_payment
            ),
            delta=format_delta(
                digital_payment_delta
            ),
            help=(
                "Share of adults who made or received a "
                "digital payment."
            ),
        )

    with metric_4:
        policy_target = 60.0

        if latest_account is None:
            target_gap = None
        else:
            target_gap = (
                policy_target
                - latest_account
            )

        st.metric(
            label="Gap to 60% Target",
            value=(
                "N/A"
                if target_gap is None
                else f"{target_gap:.1f} pp"
            ),
            help=(
                "Additional percentage points required to "
                "reach the 60% Account Ownership target."
            ),
        )

    st.divider()

    left_chart, right_chart = st.columns(
        [1.2, 1]
    )

    with left_chart:
        st.subheader(
            "Account Ownership Trajectory"
        )

        if account_ownership.empty:
            empty_chart_message(
                "The Account Ownership trajectory"
            )
        else:
            account_chart = px.line(
                account_ownership,
                x="analysis_year",
                y="value_numeric",
                markers=True,
                labels={
                    "analysis_year": "Year",
                    "value_numeric": (
                        "Adults with an account (%)"
                    ),
                },
            )

            account_chart.update_traces(
                line={
                    "width": 3,
                },
                marker={
                    "size": 9,
                },
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Account Ownership: %{y:.1f}%"
                    "<extra></extra>"
                ),
            )

            account_chart.add_hline(
                y=60,
                line_dash="dash",
                annotation_text="60% policy target",
                annotation_position="top left",
            )

            account_chart.update_layout(
                height=430,
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 20,
                    "b": 20,
                },
                hovermode="x unified",
            )

            st.plotly_chart(
                account_chart,
                use_container_width=True,
            )

            st.caption(
                "Account Ownership increased substantially "
                "between 2011 and 2021, but growth slowed "
                "between 2021 and 2024."
            )

    with right_chart:
        st.subheader(
            "Growth Between Findex Years"
        )

        usable_growth = account_growth.dropna(
            subset=["change_pp"]
        )

        if usable_growth.empty:
            empty_chart_message(
                "The Account Ownership growth chart"
            )
        else:
            growth_chart = px.bar(
                usable_growth,
                x="analysis_year",
                y="change_pp",
                text="change_pp",
                labels={
                    "analysis_year": "Survey year",
                    "change_pp": (
                        "Change from previous survey "
                        "(percentage points)"
                    ),
                },
            )

            growth_chart.update_traces(
                texttemplate="%{text:+.1f} pp",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Change: %{y:+.1f} percentage points"
                    "<extra></extra>"
                ),
            )

            growth_chart.add_hline(
                y=0,
                line_width=1,
            )

            growth_chart.update_layout(
                height=430,
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 20,
                    "b": 20,
                },
            )

            st.plotly_chart(
                growth_chart,
                use_container_width=True,
            )

            st.caption(
                "The 2021–2024 increase was only three "
                "percentage points, considerably below the "
                "gains recorded in earlier survey intervals."
            )

    st.divider()

    st.subheader(
        "Access and Digital-Finance Indicators"
    )

    indicator_frames = []

    if not account_ownership.empty:
        access_frame = account_ownership[
            [
                "analysis_year",
                "value_numeric",
            ]
        ].copy()

        access_frame["Indicator"] = (
            "Account Ownership"
        )

        indicator_frames.append(
            access_frame
        )

    if not mobile_money.empty:
        mobile_frame = mobile_money[
            [
                "analysis_year",
                "value_numeric",
            ]
        ].copy()

        mobile_frame["Indicator"] = (
            "Mobile-Money Account"
        )

        indicator_frames.append(
            mobile_frame
        )

    if not digital_payments.empty:
        usage_frame = digital_payments[
            [
                "analysis_year",
                "value_numeric",
            ]
        ].copy()

        usage_frame["Indicator"] = (
            "Digital Payment Usage"
        )

        indicator_frames.append(
            usage_frame
        )

    if indicator_frames:
        comparison_data = pd.concat(
            indicator_frames,
            ignore_index=True,
        )

        comparison_chart = px.line(
            comparison_data,
            x="analysis_year",
            y="value_numeric",
            color="Indicator",
            markers=True,
            labels={
                "analysis_year": "Year",
                "value_numeric": "Share of adults (%)",
            },
        )

        comparison_chart.update_traces(
            line={
                "width": 3,
            },
            marker={
                "size": 8,
            },
        )

        comparison_chart.update_layout(
            height=480,
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 20,
            },
            legend_title_text="Indicator",
            hovermode="x unified",
        )

        st.plotly_chart(
            comparison_chart,
            use_container_width=True,
        )
    else:
        empty_chart_message(
            "The indicator comparison"
        )

    st.divider()

    summary_left, summary_right = st.columns(2)

    with summary_left:
        st.subheader("What the data shows")

        st.markdown(
            """
            - Account Ownership rose from approximately **14% in
              2011 to 49% in 2024**.
            - The pace of measured inclusion slowed sharply after
              2021 despite rapid expansion in registered
              mobile-money accounts.
            - Registered accounts and active financial usage are
              different concepts; operator registrations should not
              be interpreted as unique active adults.
            - Usage growth depends on transactions, merchant
              acceptance, interoperability and repeated account use.
            """
        )

    with summary_right:
        st.subheader("Current analytical limitations")

        st.markdown(
            """
            - Global Findex observations are available only at
              multi-year intervals.
            - Indicators come from different sources and may use
              different definitions.
            - Operator account totals may include inactive or
              duplicate registrations.
            - Sparse observations limit causal inference and create
              wide forecasting uncertainty.
            """
        )


# -------------------------------------------------------------------
# Placeholder pages for future commits
# -------------------------------------------------------------------

elif selected_page == "Trends":
    st.header("Trends")

    st.info(
        "The interactive Trends page will be implemented "
        "in Commit 2."
    )


elif selected_page == "Event Impacts":
    st.header("Event Impacts")

    st.info(
        "The event-impact visualizations will be implemented "
        "in Commit 2."
    )


elif selected_page == "Forecasts":
    st.header("Forecasts")

    st.info(
        "The interactive forecasting page will be implemented "
        "in Commit 2."
    )


elif selected_page == "Inclusion Projections":
    st.header("Inclusion Projections")

    st.info(
        "The scenario and policy-target projections will be "
        "implemented in Commit 2."
    )


elif selected_page == "Data Explorer":
    st.header("Data Explorer")

    st.info(
        "Downloads and the full data explorer will be "
        "implemented in Commit 3."
    )