import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Prospect Value Model | Joel Hastings",
    page_icon="🏀",
    layout="wide"
)

# ── Load Artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf       = joblib.load("model_artifacts/rf_model.pkl")
    gam      = joblib.load("model_artifacts/gam_model.pkl")
    scaler   = joblib.load("model_artifacts/scaler_final.pkl")
    features = joblib.load("model_artifacts/selected_features.pkl")
    return rf, gam, scaler, features

rf, gam, scaler, features = load_artifacts()

# ── Load Draft Board ─────────────────────────────────────────
@st.cache_data
def load_draft_board():
    return pd.read_csv("model_artifacts/draft_board_2026.csv")

df_board = load_draft_board()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏀 NBA Prospect Value Model")
st.sidebar.markdown("Predicts NBA Years 3-5 WS/48 from college stats.")
page = st.sidebar.radio("Navigate", [
    "2026 Draft Board",
    "Score a Prospect",
    "About the Model"
])

# ── Page 1: Draft Board ──────────────────────────────────────
if page == "2026 Draft Board":
    st.title("2026 NBA Draft Board")
    st.markdown("Players ranked by predicted NBA Years 3-5 value.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Biggest Steal", "Tarris Reed Jr.", "Pick 26 → Model Rank 2")
    col2.metric("Biggest Reach", "AJ Dybantsa", "Pick 1 → Model Rank 19")
    col3.metric("League Average WS/48", "0.099", "")

    st.dataframe(
        df_board[[
            "player_name", "draft_pick", "model_rank",
            "draft_model_gap", "value_label",
            "predicted_ws48", "vs_average"
        ]].sort_values("model_rank"),
        use_container_width=True
    )

# ── Page 2: Score a Prospect ─────────────────────────────────
elif page == "Score a Prospect":
    st.title("Score Any Prospect")
    st.markdown("Enter college stats to get a predicted NBA WS/48.")

    col1, col2, col3 = st.columns(3)
    blk  = col1.number_input("Block % (blk_pct) — % of opponent 2PA blocked", min_value=0.0, max_value=20.0, value=3.0)
    obpm = col2.number_input("Offensive BPM (OBPM) — offensive points above average per 100 poss", min_value=-5.0, max_value=15.0, value=2.0)
    mid  = col3.number_input("Mid-Range Shots Made — total mid-range field goals made", min_value=0.0, max_value=100.0, value=20.0)

    st.caption("All stats from Bart Torvik: barttorvik.com — search any player to find these values.")
    
    if st.button("Predict"):
        X = pd.DataFrame([[blk, obpm, mid]], columns=features)
        scaled      = scaler.transform(X)
        pred_gam    = gam.predict(scaled)[0]
        vs_avg      = pred_gam - 0.099

        st.metric("Predicted WS/48", f"{pred_gam:.3f}",
                  f"{vs_avg:+.3f} vs league average")

        if vs_avg > 0.02:
            st.success("Above average — model projects NBA contributor")
        elif vs_avg > 0:
            st.info("Slightly above average — borderline rotation player")
        else:
            st.warning("Below average — limited NBA upside per model")

# ── Page 3: About ────────────────────────────────────────────
elif page == "About the Model":
    st.title("About This Model")
    st.markdown("""
    ## NBA Prospect Value Model
    Predicts NBA Years 3-5 Win Shares per 48 minutes (WS/48)
    from three college efficiency stats.

    ## Features Used
    - **blk_pct** — Block percentage (rim protection)
    - **OBPM** — Offensive Box Plus/Minus (offensive impact)
    - **mid_made** — Mid-range shots made (scoring versatility)

    ## Model Performance
    | Metric | Ridge | RF | GAM |
    |--------|-------|----|-----|
    | Test R² | 0.226 | 0.235 | 0.243 |
    | Spearman | 0.333 | 0.390 | 0.328 |

    ## Data Sources
    - College stats: Bart Torvik
    - NBA outcomes: Basketball-Reference via Kaggle
    - Draft history: NBA API

    ## Limitations
    - College players only — no international prospects
    - Model rewards efficient big men
    - Built on 2008-2021 draft classes

    ## Contact
    Built by Joel Hastings | [GitHub](https://github.com/JHastings46)
    | [LinkedIn](https://linkedin.com/in/joel-hastings-976bb855)
    """)
