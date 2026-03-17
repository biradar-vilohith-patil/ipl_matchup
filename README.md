# 🏏 IPL Tactical Matchup Engine

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI%20Framework-red.svg)

A machine learning-powered predictive analytics engine designed to optimize T20 cricket bowling strategies. 

By analyzing over 240,000 historical IPL deliveries, this system bridges the gap between dugout intuition and predictive analytics. It evaluates real-time match context to recommend the statistically optimal bowler against a specific batsman, updated for the latest IPL 2026 mega-auction squads.

## 🚀 Overview

Modern T20 cricket is won in the matchups. This engine acts as an automated data analyst, feeding a Random Forest classifier dense statistical features to predict the exact outcome of the next delivery. It then abstracts those mathematical probabilities into broadcast-quality tactical metrics.

### Key Engineering Features

* **Contextual Feature Engineering:** Dynamically evaluates a real-time `Pressure Index` based on the current over, required run rate, and wickets fallen, actively shifting the model's predictions based on match context.
* **Synthetic Feature Blending:** Solves the sparse data problem for unplayed matchups. If a bowler faces a batsman for the first time, the pipeline dynamically blends the batsman's historical strike rate with the bowler's career economy to simulate a statistical baseline.
* **Fuzzy Name Resolution:** Implements Levenshtein distance matching (`difflib`) to seamlessly resolve inconsistent dataset nomenclatures (e.g., mapping "Piyush Chawla" to "PP Chawla") without data leakage.
* **Dimensionality Reduction:** Avoids severe OOM (Out of Memory) crashes by stripping categorical player names from the training set, forcing the model to evaluate players strictly by their continuous performance metrics.

## 🧠 How the Math Works

The engine runs a **4-Way Classification** for every available bowler in a squad, grouping historical ball-by-ball outcomes into tactical classes: `dot`, `rotation`, `boundary`, and `wicket`.

To translate these raw ML probabilities into actionable coaching decisions, the system applies a **Heuristic Weighting Formula**:
```text
Tactical Score = (Dot % * 40) + (Wicket % * 100) + (Rotation % * 15) - (Boundary % * 40)
This algorithm rigorously balances aggression against economy, rewarding high-wicket threats while severely penalizing boundary leakage.

🛠 Tech Stack
Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (Random Forest Classifier)

Frontend Application: Streamlit

Data Visualization: Plotly

Architecture: Modular Object-Oriented Pipeline (Ingestion → Transformation → Training → Inference)

💻 Installation & Usage
1. Clone the repository
```bash
git clone [https://github.com/yourusername/ipl-matchup-engine.git](https://github.com/biradar-vilohith-patil/ipl-matchup-engine.git)
cd ipl-matchup-engine
```
2. Set up the environment
Create a virtual environment and install the required dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install the local package
To resolve local src import paths, install the project in editable mode:

```
pip install -e .
4. Run the Application
Launch the interactive Streamlit dashboard:
```
```
streamlit run app.py
```
📁 Project Structure
Plaintext
├── artifacts/              # Contains trained model.pkl and preprocessor.pkl
├── data/
│   └── processed/          # Cleaned IPL ball-by-ball dataset
├── src/
│   ├── components/         # Data ingestion, transformation, and model trainer modules
│   ├── pipeline/           # Training and prediction pipelines
│   ├── exception.py        # Custom exception handling
│   └── logger.py           # Logging configuration
├── app.py                  # Streamlit frontend and UI logic
├── setup.py                # Local package setup
└── README.md