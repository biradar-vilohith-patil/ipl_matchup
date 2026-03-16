# 🏏 IPL Tactical Matchup Engine

> **ML-powered ball-by-ball intelligence for IPL tactical decisions.**  
> Find the best bowler for any batsman, in any over, under any pressure situation — instantly.

---

## 📸 Overview

The IPL Tactical Matchup Engine is a machine learning application that predicts ball-by-ball outcomes between any batsman–bowler pair in the IPL. It goes beyond simple stats — it factors in match phase, pressure index, current run rate, wickets fallen, and historical head-to-head data to surface the most tactically effective bowler from a team's current squad.

---

## ✨ Features

- 🎯 **Smart Bowler Ranking** — Scores every bowler in a team's squad against a selected batsman using a custom Tactical Score formula
- ⚡ **Pressure Index** — Combines required run rate, current run rate, and wickets fallen into a single real-time pressure signal
- 🏟️ **Phase Awareness** — Powerplay / Middle Overs / Death Overs logic built into every prediction
- 📊 **Rich Visualisations** — Grouped probability charts, radar fingerprints, ranked bowler cards
- 🚀 **Batch Inference** — All bowlers evaluated in a single model call (no per-bowler loop overhead)
- 🗓️ **Current Squad Enforcement** — Hardcoded current IPL squads prevent stale historical data from surfacing wrong players
- 💾 **Cached Pipeline** — `@st.cache_resource` and `@st.cache_data` ensure the model loads once and repeat queries are instant

---

## 🗂️ Project Structure

```
ipl_matchup/
│
├── data/
│   └── processed/
│       └── clean_ipl_data.csv          # Cleaned ball-by-ball IPL dataset
│
├── artifacts/
│   ├── model.pkl                       # Trained RandomForest / GradientBoosting model
│   └── preprocessor.pkl                # Fitted ColumnTransformer (OHE + StandardScaler)
│
├── src/
│   ├── __init__.py
│   ├── exception.py                    # Custom exception with file + line info
│   ├── logger.py                       # Timestamped log files under /logs
│   ├── utils.py                        # save_object / load_object helpers
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py           # Loads CSV, splits train/test
│   │   ├── data_transformation.py      # Feature engineering + preprocessing pipeline
│   │   └── model_trainer.py            # Multi-model training with cross-validation
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py           # End-to-end training orchestrator
│       └── predict_pipeline.py         # Inference pipeline + tactical scoring
│
├── streamlit/
│   └── app.py                          # Streamlit UI application
│
├── logs/                               # Auto-generated timestamped log files
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### Target Variable
Each ball is classified into one of 7 outcomes:

| Outcome | Description |
|---------|-------------|
| `dot` | No runs scored |
| `single` | 1 run |
| `double` | 2 runs |
| `triple` | 3 runs |
| `four` | Boundary (4) |
| `six` | Maximum (6) |
| `wicket` | Batsman dismissed |

### Features Used

**Numerical:**
- `over`, `ball` / pressure index
- `bowler_economy`, `batsman_strike_rate`
- `strike_rate_vs_bowler`, `dismissal_rate`, `avg_runs`

**Categorical:**
- `batsman`, `bowler`, `batting_team`, `bowling_team`
- `venue`, `match_phase` (powerplay / middle / death)

### Tactical Score Formula
```
Tactical Score = (0.6 × Dot%) + (1.0 × Wicket%) − (0.7 × Boundary%)
```
Higher score = more bowler-favourable matchup. Negative score = batsman dominates.

### Pressure Index Formula
```
Wicket Pressure  = wickets_fallen × 0.4          # range 0 – 3.6
RR Pressure      = min(req_rr / current_rr, 3.0) # for chases
                 = min(current_rr / 8.5, 2.0)     # for 1st innings
Pressure Index   = min(Wicket Pressure + RR Pressure, 6.0)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/ipl_matchup.git
cd ipl_matchup
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python -m src.pipeline.train_pipeline
```
This will:
- Ingest `data/processed/clean_ipl_data.csv`
- Engineer features and fit the preprocessor
- Train and evaluate multiple models (RandomForest, GradientBoosting, LogisticRegression)
- Save the best model to `artifacts/model.pkl` and `artifacts/preprocessor.pkl`

### 5. Run the Streamlit app
```bash
streamlit run streamlit/app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
joblib
```

Install all at once:
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

---

## 🔧 Configuration

### Updating IPL Squads
The app uses a `CURRENT_SQUADS` dictionary at the top of `streamlit/app.py` to ensure only real squad members appear as bowler options. **Update this each season:**

```python
CURRENT_SQUADS = {
    "Mumbai Indians": ["Jasprit Bumrah", "Hardik Pandya", ...],
    "Chennai Super Kings": ["Deepak Chahar", "Ravindra Jadeja", ...],
    # ... etc
}
```

This prevents stale historical CSV data from suggesting bowlers who no longer play for a franchise.

### Changing the Data Source
Update the path in `DataIngestionConfig` inside `src/components/data_ingestion.py`:
```python
raw_csv_path: str = os.path.join("data", "processed", "clean_ipl_data.csv")
```

---

## 📊 Model Performance

The trainer evaluates three models and automatically selects the best one:

| Model | Notes |
|-------|-------|
| `RandomForestClassifier` | `class_weight="balanced"`, `max_depth=15`, `min_samples_leaf=5` |
| `GradientBoostingClassifier` | `n_estimators=150`, `learning_rate=0.1`, `subsample=0.8` |
| `LogisticRegression` | `class_weight="balanced"`, baseline comparator |

5-fold cross-validation is run on the winning model and logged to `logs/`.

> **Note:** Ball-by-ball outcome classification is an inherently noisy problem. Class imbalance (dots dominate) is handled via `class_weight="balanced"`. Accuracy metrics alone don't tell the full story — the **ranked ordering** of bowlers by Tactical Score is the real output.

---

## 🐛 Known Issues & Notes

- **Historical head-to-head missing:** When a batsman and bowler have never faced each other in the dataset, global averages are used as fallback stats. This is expected behaviour.
- **Venue defaulting:** Venue is fetched from historical data. For new venues not in the dataset, the most common venue is used as fallback.
- **Squad accuracy:** The `CURRENT_SQUADS` dict must be manually updated each IPL season to stay accurate.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

Project Link : [Link text](https://iplmatchups.streamlit.app/)

---

<div align="center">

**Built with ❤️ for cricket analytics**  

</div>
