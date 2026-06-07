"""app/pages/01_dashboard.py

Customer churn predictions table with feature values.

Run with:
    streamlit run app/main.py
"""

import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Predictions",
    page_icon="📉",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📉 Churn Predictions")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_predictions(path: str) -> pd.DataFrame:
    """Loads and validates the predictions CSV."""
    df = pd.read_csv(path)
    required = {"customer_id", "churn_score", "churn_pred", "risk_segment"}
    missing  = required - set(df.columns)
    if missing:
        st.error(f"Missing columns in predictions file: {missing}")
        st.stop()
    return df


@st.cache_data(ttl=60)
def load_source_data(path: str) -> pd.DataFrame:
    """Loads the original dataset with customer features."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


with st.sidebar:
    predictions_path = st.text_input(
        "Predictions file",
        value="./data/predictions.csv",
    )
    source_data_path = st.text_input(
        "Source dataset",
        value="./data/Telco_customer_churn.csv",
    )

# ── Load predictions ───────────────────────────────────────────────────────────

try:
    df_pred = load_predictions(predictions_path)
except FileNotFoundError:
    st.warning("Predictions file not found. Run the pipeline first.")
    uploaded = st.file_uploader("Or upload predictions CSV", type="csv")
    if uploaded:
        df_pred = pd.read_csv(uploaded)
    else:
        st.stop()

# ── Load source data and merge ─────────────────────────────────────────────────

try:
    df_src = load_source_data(source_data_path)

    # Merge predictions with source features on customerID
    df_pred["customer_id"] = df_pred["customer_id"].astype(str)
    df_src["customerID"]   = df_src["customerID"].astype(str)
    df_full = df_pred.merge(
        df_src,
        left_on="customer_id",
        right_on="customerID",
        how="left",
    ).drop(columns=["customerID"], errors="ignore")

except FileNotFoundError:
    st.info("Source dataset not found — showing predictions only.")
    df_full = df_pred

# ── Reorder columns: prediction columns first, then features ───────────────────

pred_cols    = ["customer_id", "churn_score", "churn_pred", "risk_segment"]
feature_cols = [c for c in df_full.columns if c not in pred_cols]
df_full      = df_full[pred_cols + feature_cols]

# ── Table ──────────────────────────────────────────────────────────────────────

st.dataframe(
    df_full.rename(columns={
        "customer_id":  "Customer ID",
        "churn_score":  "Churn Score",
        "churn_pred":   "Churn Prediction",
        "risk_segment": "Risk Segment",
    }),
    use_container_width=True,
    hide_index=True,
)