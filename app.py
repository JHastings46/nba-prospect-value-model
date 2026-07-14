from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

MODEL_PATH = ARTIFACT_DIR / "rf_model.pkl"
FEATURES_PATH = ARTIFACT_DIR / "selected_features.pkl"
MEDIANS_PATH = ARTIFACT_DIR / "training_medians.pkl"
BOARD_PATH = ARTIFACT_DIR / "draft_board_2026.csv"

st.set_page_config(
    page_title="NBA Prospect Value Model | Joel Hastings",
    page_icon="🏀",
    layout="wide"
)


@st.cache_resource
def load_model_artifacts():
    try:
        rf_model = joblib.load(MODEL_PATH)
        features = list(joblib.load(FEATURES_PATH))
        medians = joblib.load(MEDIANS_PATH)
        return rf_model, features, medians
    except FileNotFoundError as e:
        st.error(
            f"Model artifact not found: {e}. Check that model_artifacts/ contains "
            "rf_model.pkl, selected_features.pkl, and training_medians.pkl, and that "
            "the filenames match exactly."
        )
        st.stop()


@st.cache_data
def load_draft_board():
    try:
        return pd.read_csv(BOARD_PATH)
    except FileNotFoundError:
        st.error(f"Draft board not found at {BOARD_PATH}.")
        st.stop()


rf_model, features, training_medians = load_model_artifacts()
df_board = load_draft_board()

st.sidebar.title("🏀 NBA Prospect Value Model")
st.sidebar.markdown("Predicts NBA Years 3-5 WS/48 from college stats.")
page = st.sidebar.radio("Navigate", [
    "2026 Draft Board",
    "Score a Prospect",
    "About the Model"
])
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Joel Hastings**")
st.sidebar.markdown(
    "[GitHub](https://github.com/JHastings46/nba-prospect-value-model.git) | "
    "[LinkedIn](https://linkedin.com/in/joel-hastings-976bb855)"
)

LEAGUE_AVG_WS48 = 0.099

if page == "2026 Draft Board":
    st.title("2026 NBA Draft Board")
    st.markdown("Players ranked by predicted NBA Years 3-5 value.")
    st.caption(
        "Draft picks shown are projected 2026 positions, not official NBA draft results."
    )

    top_prospect = df_board.loc[df_board["model_rank"].idxmin()]
    steal = df_board.loc[df_board["draft_model_gap"].idxmax()]
    reach = df_board.loc[df_board["draft_model_gap"].idxmin()]

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Top Model Prospect",
        top_prospect["player_name"],
        f"Model Rank {int(top_prospect['model_rank'])}"
    )
    col2.metric(
        "Biggest Potential Steal",
        steal["player_name"],
        f"Pick {int(steal['draft_pick'])} → Model Rank {int(steal['model_rank'])}"
    )
    col3.metric(
        "Biggest Potential Reach",
        reach["player_name"],
        f"Pick {int(reach['draft_pick'])} → Model Rank {int(reach['model_rank'])}"
    )
    st.caption(f"League average benchmark: {LEAGUE_AVG_WS48} WS/48")

    st.dataframe(
        df_board[[
            "player_name", "draft_pick", "model_rank",
            "draft_model_gap", "value_label",
            "predicted_ws48", "vs_average"
        ]].sort_values("model_rank"),
        use_container_width=True
    )

elif page == "Score a Prospect":
    st.title("Score Any Prospect")
    st.markdown("Enter college stats to get a predicted NBA WS/48.")

    col1, col2, col3, col4 = st.columns(4)
    blk = col1.number_input(
        "Block % (blk_pct) — % of opponent 2PA blocked",
        min_value=0.0, max_value=20.0, value=3.0
    )
    orb = col2.number_input(
        "Off. Rebound % (ORB_pct)",
        min_value=0.0, max_value=25.0, value=5.0
    )
    drb = col3.number_input(
        "Def. Rebound % (DRB_pct)",
        min_value=0.0, max_value=40.0, value=12.0
    )
    twop = col4.number_input(
        "2P% (twoP_per)",
        min_value=0.0, max_value=1.0, value=0.50
    )

    col5, col6, col7 = st.columns(3)
    ortg = col5.number_input(
        "Offensive Rating (ORtg)",
        min_value=70.0, max_value=140.0, value=105.0
    )
    bpm = col6.number_input(
        "BPM — overall Box Plus/Minus",
        min_value=-10.0, max_value=20.0, value=2.0
    )
    ftr = col7.number_input(
        "FTr — Free Throw Attempt Rate (FTA per FGA)",
        min_value=0.0, max_value=5.0, value=0.3
    )

    st.caption(
        "College stats used to train this model come from Kaggle user adityak2003's "
        "[college basketball advanced statistics dataset]"
        "(https://www.kaggle.com/datasets/adityak2003/college-basketball-players-20092021) "
        "(2008-2021), not College Basketball Reference directly. If you're hand-entering "
        "a prospect's stats, check that dataset's page for its original source and use "
        "the same one — stat definitions (e.g. how ORtg or BPM is calculated) can differ "
        "slightly across sites, which can shift the prediction without any visible error."
    )

    if st.button("Predict"):
        input_values = {
            "blk_pct": blk,
            "ORB_pct": orb,
            "DRB_pct": drb,
            "twoP_per": twop,
            "ORtg": ortg,
            "BPM": bpm,
            "ftr": ftr,
        }
        X = pd.DataFrame([input_values])[features]
        predicted_ws48 = rf_model.predict(X)[0]
        vs_avg = predicted_ws48 - LEAGUE_AVG_WS48

        st.metric(
            "Predicted WS/48", f"{predicted_ws48:.3f}",
            f"{vs_avg:+.3f} vs league average"
        )
        st.info(
            "This model predicts Years 3-5 NBA value — not rookie year performance. "
            "Year 1 WS/48 is typically lower as players adjust to the NBA."
        )
        if vs_avg > 0.02:
            st.success("Above average — model projects NBA contributor")
        elif vs_avg > 0:
            st.info("Slightly above average — borderline rotation player")
        else:
            st.warning("Below average — limited NBA upside per model")

elif page == "About the Model":
    st.title("About This Model")
    st.markdown(f"""
    ## NBA Prospect Value Model
    An independently designed, built, and deployed portfolio project — not a college
    assignment or university-sponsored project. Predicts NBA Years 3-5 Win Shares per
    48 minutes (WS/48) from seven pre-draft college statistics.

    ## Features Used
    | Feature | Meaning |
    |---|---|
    | `blk_pct` | Shot-blocking and rim protection |
    | `ORB_pct` | Offensive rebounding |
    | `DRB_pct` | Defensive rebounding |
    | `twoP_per` | Two-point shooting efficiency |
    | `ORtg` | Individual offensive efficiency |
    | `BPM` | Overall box-score impact |
    | `ftr` | Free-throw generation |

    ## Model Performance
    | Model | Train R² | Test R² | Train MAE | Test MAE | Spearman | P-value |
    |---|---|---|---|---|---|---|
    | Linear Regression | 0.252 | 0.192 | 0.0318 | 0.0349 | 0.367 | 0.004 |
    | Ridge Regression | 0.249 | 0.194 | 0.0320 | 0.0346 | 0.364 | 0.004 |
    | **Random Forest (selected)** | **0.343** | **0.223** | **0.0306** | **0.0347** | **0.405** | **0.001** |
    | GAM | 0.246 | 0.205 | 0.0319 | 0.0355 | 0.369 | 0.004 |

    Random Forest was selected for the highest Test R², the strongest Spearman rank
    correlation, and Test MAE within 0.0001 of Ridge — the best overall balance of
    accuracy and prospect-ranking ability, which is the priority for a draft board.

    Training used chronological draft classes 2008-2019 (305 players); testing used
    2020-2021 (60 players), so the model is evaluated on draft classes it never saw
    during training.

    ## Data Sources
    | Source | Data | Coverage |
    |---|---|---|
    | [Kaggle — adityak2003](https://www.kaggle.com/datasets/adityak2003/college-basketball-players-20092021) | College advanced statistics | 2008-2021 |
    | [Kaggle — sumitrodatta](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) | NBA advanced statistics and WS/48 | 2011-2026 |
    | NBA API | NBA draft history | 2008-2021 |
    | [Kaggle — tymoteuszdobrucki](https://www.kaggle.com/datasets/tymoteuszdobrucki/nba-anthropometric) | NBA Combine measurements | 2000-2023 |
    | 2026 college statistics file | Current prospect features | 2025-2026 season |

    ## Limitations
    - College players only — international prospects without comparable college
      statistics cannot currently be evaluated
    - Reflects box-score production only — not scouting, medical evaluation,
      interviews, film analysis, or athletic testing
    - Built and validated on 2008-2021 draft classes
    - 2026 draft positions shown are projected, not official NBA draft results
    - This is a decision-support tool, not a replacement for professional scouting

    ## Contact
    Built by Joel Hastings |
    [GitHub](https://github.com/JHastings46/nba-prospect-value-model.git) |
    [LinkedIn](https://linkedin.com/in/joel-hastings-976bb855)
    """)
