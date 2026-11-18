import streamlit as st
import pandas as pd

from utils.dashboard.chart_validator import (
    get_numeric_columns,
    get_categorical_columns,
    AGGREGATIONS,
)

from utils.dashboard.chart_renderer import render_chart
from utils.dashboard.chart_recommender import recommend_chart

# Dynamic Chart Builder


def dynamic_chart_builder(df: pd.DataFrame):


    # Validate Dataset
    if df is None or df.empty:
        st.warning("No data available for chart generation.")
        return

    # Get Columns
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)
    all_columns = categorical_columns + numeric_columns

    if not all_columns:
        st.warning("No columns available in the dataset.")
        return

    # Chart Types
    chart_types = [
        "Bar Chart",
        "Horizontal Bar",
        "Line Chart",
        "Scatter Plot",
        "Histogram",
        "Box Plot",
        "Violin Plot",
        "Pie Chart",
        "Donut Chart",
        "Area Chart",
        "Correlation Heatmap"
    ]

    # Chart Configuration
    st.markdown("<h2><b>Chart Configuration</b></h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Chart Type
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            chart_types,
            key="chart_builder_type"
        )

    # X Axis
    x_axis = None
    with col2:
        if chart_type == "Correlation Heatmap":
            st.info("All numerical features are analyzed automatically.")
            x_axis = None
        else:
            if chart_type == "Histogram":
                x_options = numeric_columns
            elif chart_type in ["Pie Chart", "Donut Chart"]:
                x_options = categorical_columns
            elif chart_type == "Scatter Plot":
                x_options = numeric_columns
            else:
                x_options = all_columns

            if not x_options:
                st.warning("No suitable columns available for X Axis.")
                return

            default_x_idx = 0
            if "Category" in x_options and chart_type in ["Bar Chart", "Horizontal Bar", "Box Plot", "Violin Plot", "Pie Chart", "Donut Chart"]:
                default_x_idx = x_options.index("Category")
            elif "Temperature" in x_options and chart_type in ["Histogram", "Scatter Plot"]:
                default_x_idx = x_options.index("Temperature")

            x_axis = st.selectbox(
                "X Axis",
                x_options,
                index=default_x_idx,
                key="chart_builder_x"
            )

    # Y Axis
    y_axis = None
    with col3:
        y_chart_types = [
            "Bar Chart",
            "Horizontal Bar",
            "Line Chart",
            "Scatter Plot",
            "Box Plot",
            "Violin Plot",
            "Area Chart",
        ]

        if chart_type in y_chart_types:
            y_options = [
                col
                for col in numeric_columns
                if col != x_axis
            ]

            if not y_options:
                st.warning("No suitable numeric column available for Y Axis.")
                return

            default_y_idx = 0
            if "Shelf_Life" in y_options:
                default_y_idx = y_options.index("Shelf_Life")
            elif "Quality_Score" in y_options:
                default_y_idx = y_options.index("Quality_Score")

            y_axis = st.selectbox(
                "Y Axis",
                y_options,
                index=default_y_idx,
                key="chart_builder_y"
            )
        else:
            st.info("Y Axis is not required for this chart.")

    st.divider()

    # Aggregation & Color
    c_agg, c_col = st.columns(2)

    aggregation = None
    with c_agg:
        if chart_type in [
            "Bar Chart",
            "Horizontal Bar",
            "Line Chart",
            "Area Chart",
        ]:
            aggregation = st.selectbox(
                "Aggregation",
                list(AGGREGATIONS.keys()),
                key="chart_builder_aggregation"
            )
        else:
            st.caption("Aggregation not applicable for this chart type.")

    with c_col:
        color_options = ["None"] + [c for c in categorical_columns if c != x_axis]
        default_color_idx = 0
        if "Spoilage_Risk" in color_options and chart_type in ["Scatter Plot", "Box Plot", "Violin Plot", "Histogram"]:
            default_color_idx = color_options.index("Spoilage_Risk")

        color = st.selectbox(
            "Color Grouping",
            color_options,
            index=default_color_idx,
            key="chart_builder_color"
        )
        if color == "None":
            color = None

    st.divider()

    # Chart Recommendation
    if x_axis is not None:
        try:
            recommendation = recommend_chart(
                df=df,
                x_column=x_axis,
                y_column=y_axis
            )
            if recommendation:
                recommended_chart = recommendation.get("chart", "Not available")
                reason = recommendation.get("reason", "")
                st.info(f"**Recommended Chart:** {recommended_chart}  |  *Reason:* {reason}")
        except Exception:
            pass

    # Chart Appearance
    col_theme, col_height = st.columns(2)
    with col_theme:
        theme = st.selectbox(
            "Theme",
            [
                "plotly",
                "plotly_white",
                "plotly_dark",
                "ggplot2",
                "presentation",
                "simple_white",
            ],
            key="chart_builder_theme"
        )

    with col_height:
        height = st.slider(
            "Chart Height",
            min_value=400,
            max_value=900,
            value=550,
            step=50,
            key="chart_builder_height"
        )

    st.divider()

    # Generate Chart Button
    generate_chart = st.button(
        "Generate Selected Chart",
        use_container_width=True,
        key="generate_selected_chart"
    )

    # Initial default generation if empty
    should_generate = generate_chart or ("eda_chart_fig" not in st.session_state)

    if should_generate:
        if chart_type != "Correlation Heatmap":
            if not x_axis:
                st.error("Please select an X Axis.")
                return

            if chart_type in y_chart_types and not y_axis:
                st.error("Please select a Y Axis.")
                return

        with st.spinner("Generating chart..."):
            try:
                fig = render_chart(
                    df=df,
                    chart_type=chart_type,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    aggregation=aggregation,
                    color=color,
                    theme=theme,
                    height=height,
                )
                st.session_state["eda_chart_fig"] = fig
                st.session_state["eda_chart_success"] = True
                st.session_state["eda_chart_error"] = None
            except Exception as e:
                st.session_state["eda_chart_fig"] = None
                st.session_state["eda_chart_success"] = False
                st.session_state["eda_chart_error"] = e

    # Display Chart
    if st.session_state.get("eda_chart_error") is not None:
        st.error("Unable to generate the selected chart.")
        st.exception(st.session_state["eda_chart_error"])

    elif st.session_state.get("eda_chart_success") and st.session_state.get("eda_chart_fig") is not None:
        st.plotly_chart(
            st.session_state["eda_chart_fig"],
            use_container_width=True
        )
