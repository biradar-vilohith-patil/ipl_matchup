# 🏏 IPL Tactical Matchup Engine

A machine learning-powered predictive analytics engine designed to optimize T20 cricket bowling strategies.

By analyzing over **240,000 historical IPL deliveries**, this system bridges the gap between dugout intuition and predictive analytics. It evaluates real-time match context to recommend the statistically optimal bowler against a specific batsman, updated for the latest IPL 2026 mega-auction squads.

---

## 🚀 Overview

Modern T20 cricket is won in the matchups. This engine acts as an automated data analyst, feeding a Random Forest classifier dense statistical features to predict the exact outcome of the next delivery. It then abstracts those mathematical probabilities into broadcast-quality tactical metrics.

---

## ⚙️ Key Engineering Features

* **Contextual Feature Engineering:** Dynamically evaluates a real-time `Pressure Index` based on the current over, required run rate, and wickets fallen, actively shifting the model's predictions based on match context.

* **Synthetic Feature Blending:** Solves the sparse data problem for unplayed matchups. If a bowler faces a batsman for the first time, the pipeline dynamically blends the batsman's historical strike rate with the bowler's career economy to simulate a statistical baseline.

* **Fuzzy Name Resolution:** Implements Levenshtein distance matching (`difflib`) to seamlessly resolve inconsistent dataset nomenclatures (e.g., mapping "Piyush Chawla" to "PP Chawla") without data leakage.

* **Dimensionality Reduction:** Avoids severe OOM (Out of Memory) crashes by stripping categorical player names from the training set, forcing the model to evaluate players strictly by their continuous performance metrics.

---

## 🧠 How the Math Works

The engine runs a **4-Way Classification** for every available bowler in a squad, grouping historical ball-by-ball outcomes into tactical classes:

* `dot`
* `rotation`
* `boundary`
* `wicket`

### 📊 Tactical Scoring Formula

```text
Tactical Score = (Dot % * 40) + (Wicket % * 100) + (Rotation % * 15) - (Boundary % * 40)
```

This heuristic balances aggression against economy, rewarding wicket-taking ability while penalizing boundary leakage.

---

## 🛠 Tech Stack

* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Frontend Application:** Streamlit
* **Data Visualization:** Plotly
* **Architecture:** Modular Object-Oriented Pipeline

  * Ingestion → Transformation → Training → Inference

---

## 💻 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/biradar-vilohith-patil/ipl-matchup-engine.git
cd ipl-matchup-engine
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Local Package

```bash
pip install -e .
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```plaintext
├── artifacts/              # Trained model and preprocessor
├── data/
│   └── processed/          # Cleaned IPL dataset
├── src/
│   ├── components/         # Data ingestion, transformation, model trainer
│   ├── pipeline/           # Training & prediction pipelines
│   ├── exception.py        # Custom exception handling
│   └── logger.py           # Logging configuration
├── app.py                  # Streamlit UI
├── setup.py                # Package setup
└── README.md
```

---

## 📌 Notes

* Designed to work with IPL ball-by-ball datasets
* Supports unseen player matchups via synthetic blending
* Optimized for low-memory environments

##
---
    **Built for Cricket Learners**
