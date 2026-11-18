import streamlit as st
import pandas as pd
import importlib
from pathlib import Path
import sys

# Ensure the project root is on the import path even when Streamlit is started
# outside the repository's root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

dynamic_chart_builder = importlib.import_module(
    "utils.dashboard.chart_builder"
).dynamic_chart_builder


# Dynamic Chart Builder & EDA Page


def chart_builder_page():
    st.subheader("Dynamic Chart Builder & Exploratory Data Analysis")

    st.markdown(
        "Interactive exploratory data analysis and visualization of produce characteristics, "
        "environmental IoT sensor streams, storage parameters, and spoilage risk factors."
    )
    st.divider()

    # Load Dataset
    dataset_path = Path("dataset") / "Produce_Dataset.csv"

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        st.error(f"Dataset Loading Error: {e}")
        return

    if df is None or df.empty:
        st.warning("Produce dataset is empty or unavailable.")
        return

    # Dynamic Custom Chart Builder
    dynamic_chart_builder(df)

    st.divider()
    st.caption("All charts dynamically populated from the active Produce Dataset.")
