import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path


# Paths 

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Load Spoilage Risk (Classification) Model 

def _load_spoilage_model():
    """Load spoilage risk classification pipeline."""
    # Priority: spoilage_risk_pipeline.pkl → best_model.pkl
    for name in ["spoilage_risk_pipeline.pkl", "best_model.pkl"]:
        path = MODEL_DIR / name
        if path.exists():
            return joblib.load(path), path
    raise FileNotFoundError(
        "No spoilage risk model found in models/ folder. "
        "Please run the Jupyter notebook (post_harvest_ml_pipeline.ipynb) "
        "to train and save the model."
    )

model, _model_path = _load_spoilage_model()


# Load Shelf Life (Regression) Model 

def _load_shelf_life_model():
    """Load shelf life regression pipeline (optional)."""
    path = MODEL_DIR / "shelf_life_pipeline.pkl"
    if path.exists():
        return joblib.load(path)
    return None

shelf_life_model = _load_shelf_life_model()

# Feature Columns 

# Extract expected feature column names directly from the Pipeline model
if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    feature_columns = [
        "Category", "Produce_Name", "Variety", "Produce_Grade", "Quality_Score",
        "Health_Score", "Market_Price", "Transport_Mode", "Distance_to_Market",
        "Cold_Chain_Available", "Sensor_Battery_Level", "GPS_Latitude", "GPS_Longitude",
        "Market_Demand", "Storage_Cost", "State", "District", "Storage_Type",
        "Package_Type", "Storage_Duration", "Temperature", "Humidity", "Light_Intensity",
        "CO2_Level", "Ethylene_Level", "Moisture_Content", "Weight", "Weather",
        "Season", "Disease_Detected", "Sensor_Status"
    ]


# Model Name Normalizer 

def normalize_model_name(name):
    """
    Standardize model name to its exact original algorithm name.
    Preserves exact original algorithm names:
      - Decision Tree
      - Random Forest
      - Logistic Regression
      - XGBoost
      - Gradient Boosting
    Removes dynamically changing prefixes, suffixes, or abbreviations.
    """
    if not name or not isinstance(name, str):
        return "Decision Tree"

    clean = name.strip()
    lower = clean.lower()

    if "decision tree" in lower or lower == "dt":
        return "Decision Tree"
    elif "random forest" in lower or lower == "rf":
        return "Random Forest"
    elif "logistic" in lower or lower == "lr":
        return "Logistic Regression"
    elif "xgboost" in lower or lower == "xgb":
        return "XGBoost"
    elif "gradient boosting" in lower or lower == "gb":
        return "Gradient Boosting"

    # Remove generic wrappers if present
    for suffix in [" Classifier", " Regressor", " Pipeline", " Model", " default", " Default"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)].strip()
    if clean.startswith("Default "):
        clean = clean[len("Default "):].strip()

    return clean


# Load Model Info 

def get_model_info():
    """
    Return model metadata from active_model.json and training_report.json.
    Safe to call even if files are missing.
    """
    info = {
        "model_name": "Decision Tree",
        "model_file": str(_model_path.name),
        "accuracy": None,
        "trained_at": None,
        "source": "post_harvest_ml_pipeline.ipynb",
        "shelf_life_model_available": shelf_life_model is not None,
    }

    # Try active_model.json
    active_path = MODEL_DIR / "active_model.json"
    if active_path.exists():
        try:
            with open(active_path, "r") as f:
                data = json.load(f)
            info.update({
                "model_name": normalize_model_name(data.get("best_model", info["model_name"])),
                "accuracy": data.get("accuracy"),
                "trained_at": data.get("trained_at"),
                "source": data.get("pipeline_source", info["source"]),
            })
        except Exception:
            pass

    # Try training_report.json for extra metrics
    report_path = MODEL_DIR / "training_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r") as f:
                report = json.load(f)
            info.update({
                "precision": report.get("precision"),
                "recall": report.get("recall"),
                "f1_score": report.get("f1_score"),
                "dataset_rows": report.get("dataset_rows"),
                "features_used": report.get("features_used"),
                "shelf_life_r2": report.get("shelf_life_metrics", {}).get("r2_score"),
                "shelf_life_mae": report.get("shelf_life_metrics", {}).get("mae_days"),
            })
        except Exception:
            pass

    return info


#  Helper: Prepare DataFrame for Pipeline 

def _prepare_dataframe(df):
    """
    Ensures input dataframe contains all required features aligned to the
    Pipeline model. Handles column name normalization and missing feature
    backfilling.
    """
    df = df.copy()

    # Column header normalization & mapping
    normalized_existing = {
        str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }

    rename_map = {}
    for feature in feature_columns:
        norm_feat = feature.lower().replace(" ", "_").replace("-", "_")
        if feature not in df.columns and norm_feat in normalized_existing:
            rename_map[normalized_existing[norm_feat]] = feature

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure all required feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]


#  Helper: Risk Level Mapping 

def _get_risk_level(prediction_text, confidence=None):
    """Maps model output class to risk level string."""
    label = str(prediction_text).strip()
    label_lower = label.lower()

    if label_lower in ["high", "medium", "low"]:
        return label.capitalize()
    elif "spoil" in label_lower or "bad" in label_lower or "critical" in label_lower:
        return "High"
    elif "fresh" in label_lower or "good" in label_lower or "normal" in label_lower:
        return "Low"
    elif "warning" in label_lower or "moderate" in label_lower:
        return "Medium"

    return label


# Single Prediction 

def predict(input_data):
    """
    Generates spoilage risk prediction for a single record dict using the
    Scikit-Learn Pipeline trained in post_harvest_ml_pipeline.ipynb.

    Returns:
        (prediction_text, confidence, risk_level)
    """
    input_df = pd.DataFrame([input_data])
    prepared_df = _prepare_dataframe(input_df)

    prediction = model.predict(prepared_df)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prepared_df)[0]
        confidence = float(np.max(probabilities) * 100)
    else:
        confidence = 100.0

    prediction_text = str(prediction)
    risk_level = _get_risk_level(prediction_text, confidence)

    return (prediction_text, confidence, risk_level)


# Shelf Life Prediction 

def predict_shelf_life(input_data):
    """
    Predicts shelf life (in days) for a single record dict using the
    regression pipeline trained in post_harvest_ml_pipeline.ipynb.

    Returns:
        float (predicted days) or None if model unavailable
    """
    if shelf_life_model is None:
        return None

    input_df = pd.DataFrame([input_data])
    prepared_df = _prepare_dataframe(input_df)

    try:
        days = float(shelf_life_model.predict(prepared_df)[0])
        return round(days, 1)
    except Exception:
        return None


#  Bulk Prediction 

def predict_bulk(input_df):
    """
    Generates batch predictions for a DataFrame using the Scikit-Learn Pipeline
    trained in post_harvest_ml_pipeline.ipynb.

    Returns:
        result_df with columns: Prediction, Confidence, Risk Level,
        and optionally Predicted_Shelf_Life_Days
    """
    if input_df is None or input_df.empty:
        raise ValueError("Input dataframe is empty.")

    original_df = input_df.copy()
    prepared_df = _prepare_dataframe(original_df)

    predictions = model.predict(prepared_df)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prepared_df)
        confidence_values = np.max(probabilities, axis=1) * 100
    else:
        confidence_values = np.full(len(predictions), 100.0)

    prediction_labels = [str(val) for val in predictions]
    risk_levels = [
        _get_risk_level(pred, conf)
        for pred, conf in zip(prediction_labels, confidence_values)
    ]

    result_df = original_df.copy()
    result_df["Prediction"] = prediction_labels
    result_df["Confidence"] = np.round(confidence_values, 2)
    result_df["Risk Level"] = risk_levels

    # Add shelf life predictions if model available
    if shelf_life_model is not None:
        try:
            shelf_days = shelf_life_model.predict(prepared_df)
            result_df["Predicted_Shelf_Life_Days"] = np.round(shelf_days, 1)
        except Exception:
            pass

    return result_df


# Bulk PDF Report Generator 

def generate_bulk_prediction_pdf(result_df, prediction_time="", title="Bulk Prediction Analysis Report"):
    """
    Generate a clean, professional PDF report from prediction results or history records.
    Uses reportlab to generate the PDF in memory.
    """
    import io
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    if result_df is None or result_df.empty:
        raise ValueError("No records available to generate PDF report.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=20,
        rightMargin=20,
        topMargin=20,
        bottomMargin=20
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1E3A8A"),
        alignment=1
    )
    sub_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        alignment=1
    )

    story = [
        Paragraph("Post-Harvest Loss Prevention System", title_style),
        Spacer(1, 2),
        Paragraph(title, sub_style),
    ]
    if prediction_time:
        story.append(Paragraph(f"Generated On: {prediction_time}", meta_style))
    story.append(Spacer(1, 8))

    # Summary Metrics Card
    total = len(result_df)
    risk_col = next((c for c in ["Risk Level", "risk_level"] if c in result_df.columns), None)
    low = (result_df[risk_col].astype(str).str.lower() == "low").sum() if risk_col else 0
    med = (result_df[risk_col].astype(str).str.lower() == "medium").sum() if risk_col else 0
    high = (result_df[risk_col].astype(str).str.lower() == "high").sum() if risk_col else 0

    conf_col = next((c for c in ["Confidence", "confidence"] if c in result_df.columns), None)
    avg_conf = (
        pd.to_numeric(result_df[conf_col], errors="coerce").mean()
        if conf_col else 0.0
    )

    metrics_text = (
        f"<b>Summary Metrics:</b> Total Records: {total}   |   "
        f"Low Risk: {low}   |   Medium Risk: {med}   |   High Risk: {high}   |   "
        f"Avg Confidence: {avg_conf:.2f}%"
    )
    metric_p = Paragraph(
        metrics_text,
        ParagraphStyle("MText", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#1E293B"))
    )
    metric_table = Table([[metric_p]], colWidths=[790])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(metric_table)
    story.append(Spacer(1, 8))

    # Determine headers and data
    headers = ["#", "Produce Name", "Category", "Prediction", "Risk Level", "Confidence", "Model / Details"]
    col_widths = [30, 150, 120, 110, 85, 85, 210]
    table_data = [headers]

    for idx, (_, row) in enumerate(result_df.iterrows()):
        p_name = str(row.get("Produce_Name", row.get("produce_name", "N/A")))[:30]
        cat = str(row.get("Category", row.get("category", "N/A")))[:25]
        pred = str(row.get("Prediction", row.get("prediction", "N/A")))[:25]
        risk = str(row.get("Risk Level", row.get("risk_level", "N/A")))

        c_val = row.get("Confidence", row.get("confidence", 0))
        try:
            conf_str = f"{float(c_val):.2f}%"
        except Exception:
            conf_str = str(c_val)

        # Contextual extra info (exact model name or shelf life)
        raw_model = row.get("model_name", row.get("Model Name", None))
        shelf = row.get("Shelf Life", row.get("Predicted_Shelf_Life_Days", None))
        if raw_model:
            extra_info = normalize_model_name(str(raw_model))
        elif shelf is not None:
            extra_info = f"Shelf Life: {shelf}"
        else:
            extra_info = "Decision Tree"

        table_data.append([
            str(idx + 1),
            p_name,
            cat,
            pred,
            risk,
            conf_str,
            extra_info
        ])

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 1), (3, -1), "LEFT"),
        ("ALIGN", (6, 1), (6, -1), "LEFT"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]
    for r in range(1, len(table_data)):
        bg = colors.HexColor("#F8FAFC") if r % 2 == 1 else colors.white
        t_style.append(("BACKGROUND", (0, r), (-1, r), bg))
    table.setStyle(TableStyle(t_style))
    story.append(table)

    doc.build(story)
    return buf.getvalue()

