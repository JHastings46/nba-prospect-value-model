

# NBA Prospect Value Model

### A Pre-Draft Analytics Tool for Front-Office Decision-Making

Built by **Joel Hastings** | M.S. Data Science
[Live App](https://nba-prospect-value-model-jzpnetqjm2cr2wkgjhcrze.streamlit.app/) | [LinkedIn](https://linkedin.com/in/joel-hastings-976bb855)

---

## The Problem

NBA draft evaluations often place heavy weight on scoring volume, draft reputation, and traditional counting statistics.

This project tests whether college efficiency and impact metrics can help identify prospects whose long-term NBA value may be overlooked because of their role, usage, or playing system.

The model predicts each prospect’s NBA value during Years 3–5 and produces an alternative draft ranking that can be compared with projected draft position.

---

## Project Goal

The model answers two front-office questions:

1. Which pre-draft college statistics are most useful for predicting NBA production?
2. Which prospects may be undervalued or overvalued relative to their projected draft position?

The target variable is:

```python
ws48_yr3_5
```

This represents a player’s minutes-weighted **Win Shares per 48 minutes during NBA Years 3–5**.

---

## Final Feature Set

LassoCV feature selection followed by VIF multicollinearity cleanup reduced the original 61 numeric features to seven final predictors:

| Feature    | What It Measures                        |
| ---------- | --------------------------------------- |
| `blk_pct`  | Shot-blocking and rim protection        |
| `ORB_pct`  | Offensive rebounding                    |
| `DRB_pct`  | Defensive rebounding                    |
| `twoP_per` | Two-point shooting efficiency           |
| `ORtg`     | Individual offensive efficiency         |
| `BPM`      | Overall box-score impact                |
| `ftr`      | Ability to generate free-throw attempts |

The final feature set emphasizes interior efficiency, rebounding, rim protection, overall production, and free-throw pressure.

---

## Model Performance

Four regression models were evaluated on the unseen 2020–2021 draft classes.

| Model             |   Test R² |   Test MAE |  Spearman |   P-value |
| ----------------- | --------: | ---------: | --------: | --------: |
| Linear Regression |     0.192 |     0.0349 |     0.367 |     0.004 |
| Ridge Regression  |     0.194 | **0.0346** |     0.364 |     0.004 |
| Random Forest     | **0.223** |     0.0347 | **0.405** | **0.001** |
| GAM               |     0.205 |     0.0355 |     0.369 |     0.004 |

### Selected Model: Random Forest

Random Forest was selected as the final model because it achieved:

* The highest Test R²
* The strongest Spearman rank correlation
* A Test MAE only `0.0001` higher than Ridge Regression
* A statistically significant ranking relationship with `p = 0.001`

For a draft-board tool, correctly ranking prospects is particularly important. Random Forest provided the strongest overall balance between prediction accuracy and rank-order performance.

---

## Random Forest Configuration

The final model uses:

```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=3,
    min_samples_leaf=15,
    random_state=42,
    n_jobs=-1
)
```

These settings keep the individual trees relatively simple and require at least 15 training players in each final leaf, helping reduce overfitting.

---

## 2026 Draft Board Preview

The current draft board ranks prospects by predicted NBA Years 3–5 WS/48.

The project uses `0.099 WS/48` as its comparison benchmark.

### Highest-Ranked Prospects

| Player            | Projected Pick | Model Rank | Predicted WS/48 |
| ----------------- | -------------: | ---------: | --------------: |
| Cameron Boozer    |              3 |          1 |           0.142 |
| Morez Johnson Jr. |              9 |          2 |           0.137 |
| Caleb Wilson      |              4 |          3 |           0.137 |
| Zuby Ejiofor      |             23 |          4 |           0.137 |
| Tarris Reed Jr.   |             26 |          5 |           0.136 |

### Potentially Undervalued Prospects

| Player           | Projected Pick | Model Rank | Gap |
| ---------------- | -------------: | ---------: | --: |
| Tarris Reed Jr.  |             26 |          5 | +21 |
| Zuby Ejiofor     |             23 |          4 | +19 |
| Joshua Jefferson |             28 |         13 | +15 |
| Cameron Carr     |             24 |         11 | +13 |
| Alex Karaban     |             29 |         16 | +13 |

A positive gap means the model ranks the player higher than his projected draft position.

### Potentially Overvalued Prospects

| Player           | Projected Pick | Model Rank | Gap |
| ---------------- | -------------: | ---------: | --: |
| Darryn Peterson  |              2 |         25 | -23 |
| Keaton Wagler    |              5 |         24 | -19 |
| Mikel Brown Jr.  |              6 |         21 | -15 |
| Darius Acuff Jr. |              7 |         22 | -15 |
| AJ Dybantsa      |              1 |         12 | -11 |

A negative gap means the projected draft position ranks the player higher than the model.

These labels do not mean that a prospect will succeed or fail. They show where the player’s current college statistical profile differs from the projected draft market.

> **Board status:** These rankings reflect the most recent displayed 25-player run. The board should be regenerated after applying the corrected dataset names for Labaron Philon Jr. and Chris Cenac Jr., because adding them may shift the final model ranks.

---

## How the Value Labels Work

```python
def label_value(gap):
    if gap >= 5:
        return "Undervalued"
    elif gap <= -5:
        return "Overvalued"
    else:
        return "Fair Value"
```

* **Undervalued:** model ranks the prospect at least five spots higher
* **Overvalued:** model ranks the prospect at least five spots lower
* **Fair Value:** projected pick and model rank are within four spots

---

## Data Pipeline

| Source                       | Data                              | Coverage         |
| ---------------------------- | --------------------------------- | ---------------- |
| Kaggle — adityak2003         | College advanced statistics       | 2008–2021        |
| Kaggle — sumitrodatta        | NBA advanced statistics and WS/48 | 2011–2026        |
| NBA API                      | NBA draft history                 | 2008–2021        |
| Kaggle — tymoteuszdobrucki   | NBA Combine measurements          | 2000–2023        |
| 2026 college statistics file | Current prospect features         | 2025–2026 season |

---

## Methodology

* **Target:** Minutes-weighted NBA WS/48 during Years 3–5
* **Minimum NBA minutes:** 500 total minutes across Years 3–5
* **Initial numeric features:** 61
* **Final selected features:** 7
* **Missing-value treatment:** Training-data median imputation
* **Train/test split:** Chronological

  * Training classes: 2008–2019
  * Testing classes: 2020–2021
* **Feature selection:** LassoCV followed by VIF cleanup
* **Models compared:** Linear Regression, Ridge Regression, Random Forest, and GAM
* **Final model:** Random Forest
* **Primary evaluation metrics:** Test R², Test MAE, and Spearman rank correlation

A chronological split was used instead of a random split to better represent how an NBA front office would train a model on past draft classes and apply it to future prospects.

---

## Model Files

The final deployment assets include:

```text
rf_model.pkl
selected_features.pkl
training_medians.pkl
draft_board_2026.csv
```

The Random Forest uses the original unscaled seven features. The saved training medians ensure that future missing values are handled using the same information learned from the training dataset.

---

## Known Limitations

* The final model was trained primarily on college players.
* International prospects without college statistics cannot be evaluated directly.
* The model favors players with strong interior efficiency, rebounding, and rim-protection profiles.
* It does not directly measure age-adjusted development potential.
* It does not fully capture athleticism, defensive versatility, passing creativity, or perimeter shot creation.
* Medical information, injuries, team fit, and player-development environment are not included.
* WS/48 can favor efficient frontcourt players and may not capture every type of NBA contribution.
* The 2026 draft positions are manually entered projections and should not be described as actual draft results unless replaced with official selections.
* The model should support scouting decisions rather than replace film study, medical evaluation, interviews, and professional judgment.

---

## Next Steps

* Regenerate the 2026 board after resolving all college-player name differences
* Add age and class-year information
* Develop a separate evaluation process for international prospects
* Add feature-importance and SHAP explanations
* Compare model rankings with actual NBA draft outcomes
* Track Years 3–5 results as the 2026 class develops
* Deploy the final Random Forest model in Streamlit

---

# Author
**Joel Hastings**
Data Scientist | Sports Analytics
M.S. Data Science, University of Colorado Boulder

[LinkedIn](https://www.linkedin.com/in/joel-hastings-976bb855) | [GitHub Portfolio](https://github.com/JHastings46)

 
