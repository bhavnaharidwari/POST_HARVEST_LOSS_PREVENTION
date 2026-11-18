import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from typing import Optional

from utils.dashboard.chart_validator import AGGREGATIONS


# Chart Renderer


def render_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: Optional[str] = None,
    y_axis: Optional[str] = None,
    aggregation: Optional[str] = None,
    color: Optional[str] = None,
    theme: str = "plotly",
    height: int = 550,
):
    """
    Render dynamic, robust Plotly charts for Exploratory Data Analysis.
    """

    # Basic Dataset Validation
    if df is None or df.empty:
        raise ValueError("No data available for chart generation.")

    plot_df = df.copy()

    # Normalize Color
    if color is not None:
        if str(color).strip() in ["", "None"] or color not in plot_df.columns:
            color = None

    # Correlation Heatmap special handling
    if chart_type == "Correlation Heatmap":
        ignore_cols = ["Record_ID", "Farmer_ID", "Warehouse_ID", "IoT_Device_ID"]
        num_df = plot_df.select_dtypes(include="number").drop(
            columns=[c for c in ignore_cols if c in plot_df.columns], errors="ignore"
        )
        if num_df.shape[1] < 2:
            raise ValueError("At least 2 numeric columns are required for a Correlation Heatmap.")
        
        corr_matrix = num_df.corr().round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            template=theme,
            height=height
        )
        fig.update_layout(
            title=dict(text="Correlation Heatmap", x=0.5, font=dict(size=20)),
            template=theme,
            height=height,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        return fig

    # X Axis Validation
    if not x_axis:
        raise ValueError("Please select an X Axis.")

    if x_axis not in plot_df.columns:
        raise ValueError(f"X Axis column '{x_axis}' not found.")

    # Normalize Y Axis
    if y_axis is not None and str(y_axis).strip() == "":
        y_axis = None

    # Chart Types Requiring Y Axis
    y_required = [
        "Bar Chart",
        "Horizontal Bar",
        "Line Chart",
        "Scatter Plot",
        "Box Plot",
        "Violin Plot",
        "Area Chart",
    ]

    if chart_type in y_required:
        if y_axis is None:
            raise ValueError(f"Please select a Y Axis for {chart_type}.")
        if y_axis not in plot_df.columns:
            raise ValueError(f"Y Axis column '{y_axis}' not found.")

    # Numeric Validation for Y
    numeric_y_charts = [
        "Bar Chart",
        "Horizontal Bar",
        "Line Chart",
        "Scatter Plot",
        "Box Plot",
        "Violin Plot",
        "Area Chart",
    ]

    if chart_type in numeric_y_charts:
        if not pd.api.types.is_numeric_dtype(plot_df[y_axis]):
            converted = pd.to_numeric(plot_df[y_axis], errors="coerce")
            if converted.notna().sum() == 0:
                raise ValueError(f"Y Axis '{y_axis}' must contain numeric data.")
            plot_df[y_axis] = converted

    # Histogram Validation
    if chart_type == "Histogram":
        if not pd.api.types.is_numeric_dtype(plot_df[x_axis]):
            converted = pd.to_numeric(plot_df[x_axis], errors="coerce")
            if converted.notna().sum() == 0:
                raise ValueError(f"Histogram requires a numeric X Axis. '{x_axis}' is not numeric.")
            plot_df[x_axis] = converted

    # Scatter Validation
    if chart_type == "Scatter Plot":
        plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors="coerce")
        plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_axis, y_axis])
        if plot_df.empty:
            raise ValueError("No valid numeric data available for the Scatter Plot.")

    # Aggregation Handling
    aggregation_charts = [
        "Bar Chart",
        "Horizontal Bar",
        "Line Chart",
        "Area Chart",
    ]

    if aggregation and y_axis is not None and chart_type in aggregation_charts:
        if aggregation not in AGGREGATIONS:
            aggregation = "Mean"
        agg_func = AGGREGATIONS.get(aggregation, "mean")

        keep_cols = [x_axis, y_axis]
        group_cols = [x_axis]
        if color and color in plot_df.columns and color != x_axis:
            keep_cols.append(color)
            group_cols.append(color)

        temp_df = plot_df[keep_cols].copy()
        temp_df[y_axis] = pd.to_numeric(temp_df[y_axis], errors="coerce")
        temp_df = temp_df.dropna(subset=[x_axis, y_axis])

        if temp_df.empty:
            raise ValueError("No valid data available after aggregation.")

        try:
            plot_df = (
                temp_df
                .groupby(group_cols, as_index=False)[y_axis]
                .agg(agg_func)
            )
        except Exception as e:
            raise ValueError(f"Unable to apply {aggregation} aggregation: {e}")

    # PIE / DONUT Validation
    if chart_type in ["Pie Chart", "Donut Chart"]:
        if plot_df[x_axis].dropna().empty:
            raise ValueError("No valid categorical data available.")

    # Render Charts

    # Bar Chart
    if chart_type == "Bar Chart":
        fig = px.bar(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color,
            template=theme,
            text_auto=True,
            height=height,
        )

    # Horizontal Bar
    elif chart_type == "Horizontal Bar":
        fig = px.bar(
            plot_df,
            x=y_axis,
            y=x_axis,
            orientation="h",
            color=color,
            template=theme,
            text_auto=True,
            height=height,
        )

    # Line Chart
    elif chart_type == "Line Chart":
        fig = px.line(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color,
            markers=True,
            template=theme,
            height=height,
        )

    # Scatter Plot (with representative sample for responsive rendering)
    elif chart_type == "Scatter Plot":
        sample_df = plot_df.sample(min(3000, len(plot_df)), random_state=42) if len(plot_df) > 3000 else plot_df
        fig = px.scatter(
            sample_df,
            x=x_axis,
            y=y_axis,
            color=color,
            template=theme,
            height=height,
            opacity=0.75,
            trendline="ols" if color is None else None
        )

    # Histogram with Mean and Median
    elif chart_type == "Histogram":
        valid_series = plot_df[x_axis].dropna()
        fig = px.histogram(
            plot_df,
            x=x_axis,
            color=color,
            template=theme,
            height=height,
            opacity=0.8,
            marginal="box"
        )
        if not valid_series.empty:
            mean_val = float(valid_series.mean())
            median_val = float(valid_series.median())
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top left"
            )
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="#10b981",
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="top right"
            )

    # Box Plot
    elif chart_type == "Box Plot":
        sample_df = plot_df.sample(min(5000, len(plot_df)), random_state=42) if len(plot_df) > 5000 else plot_df
        fig = px.box(
            sample_df,
            x=x_axis,
            y=y_axis,
            color=color,
            template=theme,
            height=height,
            points="outliers"
        )

    # Violin Plot
    elif chart_type == "Violin Plot":
        sample_df = plot_df.sample(min(5000, len(plot_df)), random_state=42) if len(plot_df) > 5000 else plot_df
        fig = px.violin(
            sample_df,
            x=x_axis,
            y=y_axis,
            color=color,
            box=True,
            points="outliers",
            template=theme,
            height=height,
        )

    # Pie Chart
    elif chart_type == "Pie Chart":
        pie_df = (
            plot_df[x_axis]
            .dropna()
            .value_counts()
            .head(12)
            .reset_index()
        )
        pie_df.columns = [x_axis, "Count"]
        fig = px.pie(
            pie_df,
            names=x_axis,
            values="Count",
            template=theme,
            height=height,
        )

    # Donut Chart
    elif chart_type == "Donut Chart":
        donut_df = (
            plot_df[x_axis]
            .dropna()
            .value_counts()
            .head(12)
            .reset_index()
        )
        donut_df.columns = [x_axis, "Count"]
        fig = px.pie(
            donut_df,
            names=x_axis,
            values="Count",
            hole=0.55,
            template=theme,
            height=height,
        )

    # Area Chart
    elif chart_type == "Area Chart":
        fig = px.area(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color,
            markers=True,
            template=theme,
            height=height,
        )

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    # Layout Styling
    chart_title = f"{chart_type}"
    if x_axis and y_axis:
        chart_title = f"{chart_type}: {y_axis} by {x_axis}"
    elif x_axis:
        chart_title = f"{chart_type}: {x_axis}"

    fig.update_layout(
        title=dict(
            text=chart_title,
            x=0.5,
            font=dict(size=20),
        ),
        template=theme,
        height=height,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig
