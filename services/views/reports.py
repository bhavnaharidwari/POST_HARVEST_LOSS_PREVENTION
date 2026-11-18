import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("database") / "prediction_history.db"


def reports_page():

    st.title(" Prediction History")


    # Load Data


    try:

        conn = sqlite3.connect(DB_PATH)

        df = pd.read_sql_query(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            """,
            conn
        )

        conn.close()

        if not df.empty and "model_name" in df.columns:
            from utils.prediction import normalize_model_name
            df["model_name"] = df["model_name"].apply(normalize_model_name)

    except Exception as e:

        st.error(f"Database Error: {e}")
        return


    # Empty Database


    if df.empty:

        st.info("No prediction history found.")

        return

    # Search


    search = st.text_input(
        " Search Produce"
    )

    if search:

        df = df[
            df["produce_name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]


    # Category Filter


    category = st.selectbox(

        "Category",

        ["All"] + sorted(
            df["category"]
            .unique()
            .tolist()
        )

    )

    if category != "All":

        df = df[
            df["category"] == category
        ]

    st.divider()

    # Table

    st.subheader("Prediction Records")

    st.dataframe(

        df,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # Bulk Reports
    st.subheader("Bulk Reports")
    st.write(
        "Generate and download comprehensive bulk prediction and history reports in CSV or PDF format."
    )

    col1, col2 = st.columns(2)

    with col1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download Bulk Report (CSV)",
            data=csv_data,
            file_name="bulk_prediction_report.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_bulk_report_csv"
        )

    with col2:
        try:
            from utils.prediction import generate_bulk_prediction_pdf
            pdf_bytes = generate_bulk_prediction_pdf(
                df,
                title="Bulk Prediction & History Report"
            )
            st.download_button(
                label=" Download Bulk Report (PDF)",
                data=pdf_bytes,
                file_name="bulk_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_bulk_report_pdf"
            )
        except Exception as e:
            st.text(f"Unable to generate PDF report: {e}")

    st.divider()


    # Clear History


    if st.button(
        "🗑 Clear History",
        use_container_width=True
    ):

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM prediction_history"
        )

        conn.commit()

        conn.close()

        st.success(
            "Prediction history cleared successfully."
        )

        st.rerun()