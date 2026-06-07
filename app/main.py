import streamlit as st
import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.chain import build_chain

@st.cache_resource
def get_chain(model: str):
    """Loads the RAG chain — cached so it only builds once per model choice."""
    return build_chain(model=model)
 

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
    ollama_model = st.selectbox(
        "Ollama model",
        options=["llama3.2"],
        help="Must be pulled locally: ollama pull <model>",
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


# ── Chatbot ────────────────────────────────────────────────────────────────────
 
st.markdown("---")
st.subheader("🤖 Churn Assistant")
st.caption(
    "Ask me about the predictions, model, or business rules. "
)
 
 
try:
    chain, retriever = get_chain(ollama_model)
    chain_ready = True
except Exception as e:
    st.error(
        f"Could not connect to Ollama ({e}). "
        "Make sure Ollama is running and the model is pulled: "
        f"`ollama pull {ollama_model}`"
    )
    chain_ready = False
 
if chain_ready:
    if "messages" not in st.session_state:
        st.session_state.messages = []
 
    # Render conversation history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
 
    if prompt := st.chat_input("Ask a question about the churn model..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
 
        with st.spinner("Thinking..."):
            answer  = chain.invoke(prompt)
            sources = {
                doc.metadata.get("source", "unknown")
                for doc in retriever.invoke(prompt)
            }
 
        st.chat_message("assistant").write(answer)
        if sources:
            st.caption(f"Sources: {', '.join(sources)}")
 
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
        })
 