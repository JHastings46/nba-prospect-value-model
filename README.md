# NBA Prospect Value Model
### A Pre-Draft Analytics Tool for Front Office Decision Making

Built by **Joel Hastings** | MS Data Science, University of Colorado Boulder  
[Live App](https://nba-prospect-value-model-jzpnetqjm2cr2wkgjhcrze.streamlit.app/) | [LinkedIn](https://linkedin.com/in/joel-hastings-976bb855)

---

## The Problem

NBA draft boards historically overweight scoring volume 
from high-usage college players. Teams repeatedly invest 
lottery picks in high scorers whose NBA efficiency never 
matches their draft position.

This model re-ranks prospects using college efficiency 
metrics — not scoring volume — to identify players whose 
value is hidden by role, system, or low usage.

---

## The Core Finding

The draft systematically undervalues efficient big men.
Three college stats predict NBA Years 3-5 value better 
than draft position alone:

| Feature | What It Measures |
|---------|-----------------|
| blk_pct | Rim protection and defensive impact |
| BPM | Overall court impact per 100 possessions |
| ftr | Aggression and ability to draw fouls |

---

## Model Performance

| Metric | Ridge | Random Forest | GAM |
|--------|-------|---------------|-----|
| Test R² | 0.231 | 0.181 | 0.255 |
| Test MAE | 0.0342 | 0.0357 | 0.0340 |
| Spearman | 0.307 | 0.299 | 0.319 |
| P-value | 0.017 | 0.020 | 0.013 |

GAM selected as primary model. All three models 
statistically significant (p < 0.05).

---

## Validated Findings

**Correctly identified undervalued players:**

| Player | Draft Pick | Model Rank | Actual Outcome |
|--------|-----------|------------|----------------|
| Neemias Queta | 39 | 1 | #1 WS/48 in test set |
| Isaiah Jackson | 22 | 2 | #3 WS/48 in test set |
| Paul Reed | 58 | 6 | #7 WS/48 in test set |

**Correctly flagged overvalued players:**

| Player | Draft Pick | Predicted WS/48 | Actual WS/48 |
|--------|-----------|-----------------|--------------|
| Jimmer Fredette | 10 | 0.070 | 0.021 |
| Kendall Marshall | 13 | 0.067 | 0.014 |
| Cam Reddish | 10 | 0.064 | 0.030 |

---

## 2026 Draft Board Preview

Model projects these players as undervalued relative 
to their draft position:

| Player | Draft Pick | Model Rank | Gap |
|--------|-----------|------------|-----|
| Tarris Reed Jr. | 26 | 2 | +24 |
| Zuby Ejiofor | 23 | 3 | +20 |
| Joshua Jefferson | 28 | 8 | +20 |

Model flags these players as overvalued:

| Player | Draft Pick | Model Rank | Gap |
|--------|-----------|------------|-----|
| AJ Dybantsa | 1 | 20 | -19 |
| Mikel Brown Jr. | 6 | 23 | -17 |
| Darius Acuff Jr. | 7 | 25 | -18 |

---

## Data Pipeline

| Source | Data | Coverage |
|--------|------|----------|
| Kaggle (adityak2003) | College advanced stats | 2008-2021 |
| Kaggle (sumitrodatta) | NBA advanced stats / WS/48 | 2011-2026 |
| NBA API | Draft history | 2008-2021 |
| Kaggle (tymoteuszdobrucki) | Combine measurements | 2000-2023 |

---

## Methodology

- **Target variable:** Minutes-weighted WS/48, NBA Years 3-5
- **Minimum minutes:** 500 across Years 3-5
- **Train/test split:** Chronological — 2008-2019 train, 2020-2021 test
- **Feature selection:** LassoCV → VIF cleanup
- **Models compared:** Ridge, Random Forest, GAM

---

## Known Limitations

- College players only — international prospects excluded
- Model rewards frontcourt efficiency profiles
- WS/48 penalizes injury-affected seasons

---
# Author

Joel Hastings — M.S. Data Science, 
[LinkedIn](https://www.linkedin.com/in/joel-hastings-976bb855) | 
[Portfolio](https://github.com/JHastings46)
