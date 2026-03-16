import streamlit as st
import pandas as pd
import plotly.express as px

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="IPL Matchup Intelligence", layout="wide")

# ------------------------------------------------
# LOAD CSS
# ------------------------------------------------
def load_css():
    with open("streamlit/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/clean_ipl_data.csv")
    return df

df = load_data()

batsmen = sorted(df["batsman"].dropna().unique())
bowlers = sorted(df["bowler"].dropna().unique())
teams = sorted(df["batting_team"].dropna().unique())

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Matchup Analytics", "Venue Analytics"]
)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown('<div class="hero">🏏 IPL Matchup Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning powered batsman vs bowler analytics</div>', unsafe_allow_html=True)

# ------------------------------------------------
# MATCH INPUTS
# ------------------------------------------------
st.markdown("### Match Context")

col1, col2, col3 = st.columns(3)
with col1:
    batsman = st.selectbox("Batsman", batsmen)
with col2:
    bowler = st.selectbox("Bowler", bowlers)
with col3:
    batting_team = st.selectbox("Batting Team", teams)

col4, col5, col6 = st.columns(3)
with col4:
    bowling_team = st.selectbox("Bowling Team", teams)
with col5:
    over = st.slider("Over", 1, 20, 10)
with col6:
    ball = st.slider("Ball", 1, 6, 3)

# ------------------------------------------------
# MATCH PHASE
# ------------------------------------------------
def get_phase(over):
    if over <= 6:
        return "Powerplay"
    elif over <= 15:
        return "Middle Overs"
    else:
        return "Death Overs"

phase = get_phase(over)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
predictor = PredictPipeline()

# ------------------------------------------------
# PREDICTION PAGE
# ------------------------------------------------
if page == "Prediction":

    if st.button("Predict Ball Outcome"):

        data = CustomData(
            batsman=batsman,
            bowler=bowler,
            batting_team=batting_team,
            bowling_team=bowling_team,
            over=over,
            ball=ball
        )

        df_input = data.get_data_as_dataframe()

        prediction = predictor.predict(df_input)

        runs = round(prediction[0], 2)

        st.markdown("### Prediction Metrics")

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Runs", runs)
        m2.metric("Match Phase", phase)
        m3.metric("Ball", ball)

        st.write("")

        # Run distribution chart
        st.markdown("### Run Outcome Distribution")

        outcomes = [0, 1, 2, 3, 4, 6]
        probs = [0.25, 0.35, 0.15, 0.05, 0.15, 0.05]

        run_df = pd.DataFrame({
            "Runs": outcomes,
            "Probability": probs
        })

        fig = px.bar(run_df, x="Runs", y="Probability", color="Runs")
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------
# MATCHUP ANALYTICS
# ------------------------------------------------
elif page == "Matchup Analytics":

    st.markdown("### Best Bowlers vs Selected Batsman")

    results = []

    for b in bowlers:
        data = CustomData(
            batsman=batsman,
            bowler=b,
            batting_team=batting_team,
            bowling_team=bowling_team,
            over=over,
            ball=ball
        )

        df_input = data.get_data_as_dataframe()

        pred = predictor.predict(df_input)[0]

        results.append({
            "Bowler": b,
            "Expected Runs": pred
        })

    matchup_df = pd.DataFrame(results).sort_values("Expected Runs")

    best = matchup_df.head(10)

    st.dataframe(best)

    fig = px.bar(
        best,
        x="Bowler",
        y="Expected Runs",
        color="Expected Runs",
        title="Best Bowlers vs Batsman"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bowlers To Avoid")

    worst = matchup_df.tail(10).sort_values("Expected Runs", ascending=False)

    st.dataframe(worst)

    # Heatmap
    st.markdown("### Batsman vs Bowler Heatmap")

    heatmap_df = matchup_df.head(20)

    fig2 = px.imshow(
        heatmap_df[["Expected Runs"]],
        labels=dict(x="Metric", y="Bowler", color="Runs"),
        y=heatmap_df["Bowler"]
    )

    st.plotly_chart(fig2, use_container_width=True)


# ------------------------------------------------
# VENUE ANALYTICS
# ------------------------------------------------
elif page == "Venue Analytics":

    st.markdown("### Venue Scoring Trends")

    venue_df = df.groupby("venue")["batsman_runs"].mean().reset_index()

    fig3 = px.bar(
        venue_df,
        x="venue",
        y="batsman_runs",
        title="Average Runs per Ball by Venue"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Top Venues for Batting")

    best_venues = venue_df.sort_values("batsman_runs", ascending=False).head(10)

    st.dataframe(best_venues)


# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.caption("Built using Machine Learning, Streamlit and IPL Ball-by-Ball Data")