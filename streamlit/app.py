import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# ═══════════════════════════════════════════════════════
# CURRENT IPL SQUADS  (hardcoded — overrides stale CSV data)
# Update this dict each season to keep squads accurate.
# ═══════════════════════════════════════════════════════

CURRENT_SQUADS = {
    "Mumbai Indians": [
        "Jasprit Bumrah", "Hardik Pandya", "Trent Boult", "Piyush Chawla",
        "Kumar Kartikeya", "Akash Madhwal", "Jason Behrendorff", "Arjun Tendulkar",
        "Mohsin Khan", "Hrithik Shokeen", "Shams Mulani"
    ],
    "Chennai Super Kings": [
        "Deepak Chahar", "Ravindra Jadeja", "Moeen Ali", "Tushar Deshpande",
        "Matheesha Pathirana", "Simarjeet Singh", "Maheesh Theekshana",
        "Shardul Thakur", "Nishant Sindhu", "Rajvardhan Hangargekar"
    ],
    "Royal Challengers Bengaluru": [
        "Mohammed Siraj", "Glenn Maxwell", "Wanindu Hasaranga", "Harshal Patel",
        "Josh Hazlewood", "Akash Deep", "Reece Topley", "Vijaykumar Vyshak",
        "Karn Sharma", "Mayank Dagar", "Lockie Ferguson"
    ],
    "Kolkata Knight Riders": [
        "Sunil Narine", "Andre Russell", "Varun Chakaravarthy", "Mitchell Starc",
        "Harshit Rana", "Suyash Sharma", "Sakib Hussain", "Anrich Nortje",
        "Spencer Johnson", "Ramandeep Singh"
    ],
    "Delhi Capitals": [
        "Kuldeep Yadav", "Axar Patel", "Anrich Nortje", "Ishant Sharma",
        "Khaleel Ahmed", "Lungi Ngidi", "Mukesh Kumar", "Lalit Yadav",
        "Sumit Kumar", "Praveen Dubey"
    ],
    "Punjab Kings": [
        "Arshdeep Singh", "Sam Curran", "Kagiso Rabada", "Rahul Chahar",
        "Nathan Ellis", "Harpreet Brar", "Harshal Patel", "Shivam Singh",
        "Rishi Dhawan", "Vidwath Kaverappa"
    ],
    "Rajasthan Royals": [
        "Yuzvendra Chahal", "Trent Boult", "Sandeep Sharma", "Prasidh Krishna",
        "Ravichandran Ashwin", "Jason Holder", "Adam Zampa", "Kuldeep Sen",
        "Nandre Burger", "Kuldip Yadav"
    ],
    "Sunrisers Hyderabad": [
        "Bhuvneshwar Kumar", "T Natarajan", "Pat Cummins", "Mayank Markande",
        "Shahbaz Ahmed", "Marco Jansen", "Jaydev Unadkat", "Umran Malik",
        "Akeal Hosein", "Fazalhaq Farooqi"
    ],
    "Gujarat Titans": [
        "Mohammed Shami", "Rashid Khan", "Noor Ahmad", "Darshan Nalkande",
        "Mohit Sharma", "Yash Dayal", "Jayant Yadav", "Sai Kishore",
        "Kartik Tyagi", "Sandeep Warrier"
    ],
    "Lucknow Super Giants": [
        "Ravi Bishnoi", "Mohsin Khan", "Mark Wood", "Naveen-ul-Haq",
        "Yash Thakur", "Krunal Pandya", "Amit Mishra", "Kyle Mayers",
        "Shamar Joseph", "Arshad Khan"
    ],
}

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="IPL Tactical Matchup Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }

/* ── BACKGROUND ── */
.stApp {
    background: #080e14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(245,166,35,0.18) 0%, transparent 70%),
        repeating-linear-gradient(0deg, transparent, transparent 59px, rgba(255,255,255,0.025) 59px, rgba(255,255,255,0.025) 60px),
        repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(255,255,255,0.025) 59px, rgba(255,255,255,0.025) 60px);
}

/* ── HERO ── */
.hero-outer {
    position:relative; text-align:center; padding:64px 0 32px; overflow:hidden;
}
.hero-glow {
    position:absolute; top:50%; left:50%; transform:translate(-50%,-60%);
    width:700px; height:300px;
    background:radial-gradient(ellipse, rgba(245,166,35,0.22) 0%, transparent 70%);
    pointer-events:none;
}
.hero-eyebrow {
    font-size:12px; letter-spacing:5px; text-transform:uppercase;
    color:rgba(245,166,35,0.7); margin-bottom:14px;
}
.hero-title {
    font-family:'Bebas Neue',sans-serif;
    font-size:clamp(72px,11vw,140px);
    letter-spacing:8px; line-height:0.92;
    background:linear-gradient(160deg,#ffe066 0%,#f5a623 35%,#ff6b35 65%,#e8005a 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    margin:0; display:block;
}
.hero-sub {
    font-size:16px; color:rgba(255,255,255,0.45); margin-top:18px; letter-spacing:1px;
}
.hero-divider {
    width:80px; height:3px; margin:22px auto 0;
    background:linear-gradient(90deg,#f5a623,#e8005a);
    border-radius:2px;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family:'Bebas Neue',sans-serif; font-size:24px;
    letter-spacing:5px; color:#f5a623; margin:32px 0 10px;
    display:flex; align-items:center; gap:10px;
}
.section-label::after {
    content:''; flex:1; height:1px;
    background:linear-gradient(90deg,rgba(245,166,35,0.3),transparent);
}

/* ── INPUT CARD ── */
.input-card {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:18px; padding:24px 28px; margin-bottom:18px;
}

/* ── CONTEXT CARD ── */
.ctx-card {
    background:rgba(245,166,35,0.05);
    border:1px solid rgba(245,166,35,0.15);
    border-radius:16px; padding:20px 24px; margin:16px 0;
}
.ctx-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
.ctx-item { }
.ctx-lbl { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.35); margin-bottom:6px; }
.ctx-val { font-family:'Bebas Neue',sans-serif; font-size:28px; color:#fff; line-height:1; }
.ctx-val.orange { color:#f5a623; }
.ctx-val.red    { color:#e8005a; }
.ctx-val.green  { color:#64dc64; }

/* ── PRESSURE BAR ── */
.pressure-row { display:flex; align-items:center; gap:14px; margin-top:16px; }
.p-label { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.35); margin-bottom:6px; }
.p-bar-bg { flex:1; background:rgba(255,255,255,0.07); border-radius:6px; height:12px; overflow:hidden; }
.p-bar-fill { height:12px; border-radius:6px; }
.p-val { font-family:'Bebas Neue',sans-serif; font-size:26px; min-width:40px; text-align:right; }

/* ── PHASE CHIP ── */
.phase-chip { display:inline-block; padding:4px 14px; border-radius:999px; font-size:11px; letter-spacing:2px; text-transform:uppercase; font-weight:600; }
.phase-powerplay { background:rgba(100,220,100,.15); color:#64dc64; border:1px solid rgba(100,220,100,.3); }
.phase-middle     { background:rgba(245,166,35,.15);  color:#f5a623; border:1px solid rgba(245,166,35,.3); }
.phase-death      { background:rgba(232,0,90,.15);    color:#e8005a; border:1px solid rgba(232,0,90,.3); }

/* ── METRIC STRIP ── */
.metric-strip { display:flex; gap:14px; flex-wrap:wrap; justify-content:center; margin:24px 0; }
.metric-pill {
    background:rgba(245,166,35,0.07); border:1px solid rgba(245,166,35,0.2);
    border-radius:14px; padding:16px 28px; text-align:center; min-width:130px; flex:1;
}
.metric-pill .val { font-family:'Bebas Neue',sans-serif; font-size:38px; color:#f5a623; line-height:1; }
.metric-pill .lbl { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:rgba(255,255,255,0.35); margin-top:6px; }

/* ── BOWLER ROWS ── */
.bowler-row {
    display:flex; align-items:center; gap:14px;
    background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.055);
    border-radius:12px; padding:13px 18px; margin-bottom:9px;
}
.rank-badge { font-family:'Bebas Neue',sans-serif; font-size:26px; color:#f5a623; width:34px; text-align:center; flex-shrink:0; }
.bowler-name { font-size:15px; font-weight:600; color:#fff; flex:1; }
.bowler-score { font-family:'Bebas Neue',sans-serif; font-size:22px; color:#f5a623; min-width:52px; text-align:right; }
.mini-bar-wrap { flex:2; }
.mini-bar-bg { background:rgba(255,255,255,0.07); border-radius:4px; height:5px; overflow:hidden; }
.mini-bar-fill { height:5px; border-radius:4px; }

/* ── CTA BUTTON ── */
div.stButton > button {
    background:linear-gradient(135deg,#f5a623,#ff6b35) !important;
    color:#000 !important; font-family:'Bebas Neue',sans-serif !important;
    font-size:22px !important; letter-spacing:4px !important;
    border:none !important; border-radius:12px !important;
    padding:16px 0 !important; width:100% !important;
    box-shadow:0 4px 24px rgba(245,166,35,0.3) !important;
    transition:opacity .2s, transform .15s, box-shadow .2s !important;
}
div.stButton > button:hover {
    opacity:.9 !important; transform:translateY(-2px) !important;
    box-shadow:0 8px 32px rgba(245,166,35,0.45) !important;
}

/* ── DIVIDER ── */
.fancy-divider { height:1px; background:linear-gradient(90deg,transparent,rgba(245,166,35,0.35),transparent); margin:28px 0; }

/* ── FOOTER ── */
.footer { text-align:center; font-size:11px; color:rgba(255,255,255,0.2); padding:32px 0 16px; letter-spacing:3px; }

/* ── SEARCH HINT ── */
.search-hint { font-size:11px; color:rgba(255,255,255,0.3); letter-spacing:1px; margin-top:-10px; margin-bottom:8px; }

label { color:rgba(255,255,255,0.6) !important; font-size:13px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LOAD DATA + PIPELINE
# ═══════════════════════════════════════════════════════

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/clean_ipl_data.csv")

@st.cache_resource
def load_predictor():
    return PredictPipeline()

@st.cache_data
def batch_predict(bowlers: tuple, batsman, batting_team, bowling_team, over, pressure_index):
    """
    Runs all bowlers through the model in ONE batched call.
    This is the key performance fix — instead of N separate
    transform+predict calls, we do exactly 1 of each.
    """
    pipeline = load_predictor()

    rows = []
    for bowler in bowlers:
        cd = CustomData(
            batsman=batsman, bowler=bowler,
            batting_team=batting_team, bowling_team=bowling_team,
            over=over, pressure_index=pressure_index
        )
        df_row = cd.get_data_as_dataframe()

        # Enrich with historical stats (still per-bowler lookup, but cheap)
        stats = pipeline.get_stats(batsman, bowler)
        df_row["match_phase"]           = pipeline.get_phase(over)
        df_row["strike_rate_vs_bowler"] = stats["strike_rate_vs_bowler"]
        df_row["dismissal_rate"]        = stats["dismissal_rate"]
        df_row["bowler_economy"]        = stats["bowler_economy"]
        df_row["batsman_strike_rate"]   = stats["batsman_strike_rate"]
        df_row["avg_runs"]              = stats["avg_runs"]
        df_row["venue"]                 = stats["venue"]
        rows.append(df_row)

    batch_df   = pd.concat(rows, ignore_index=True)

    # ── SINGLE transform + predict call for all bowlers ──
    scaled     = pipeline.preprocessor.transform(batch_df)
    probs      = pipeline.model.predict_proba(scaled)
    classes    = pipeline.model.classes_

    results = []
    for i, bowler in enumerate(bowlers):
        prob_row = dict(zip(classes, probs[i]))

        def gp(k): return float(prob_row.get(k, 0.0))

        dot_p  = gp("dot")
        wkt_p  = gp("wicket")
        four_p = gp("four")
        six_p  = gp("six")
        bnd_p  = four_p + six_p
        score  = round((0.6 * dot_p) + (1.0 * wkt_p) - (0.7 * bnd_p), 4)

        results.append({
            "Bowler":         bowler,
            "Tactical Score": score,
            "Dot %":          round(dot_p  * 100, 1),
            "Wicket %":       round(wkt_p  * 100, 1),
            "Four %":         round(four_p * 100, 1),
            "Six %":          round(six_p  * 100, 1),
            "Boundary %":     round(bnd_p  * 100, 1),
        })

    return pd.DataFrame(results).sort_values("Tactical Score", ascending=False)


df        = load_data()
predictor = load_predictor()
all_batsmen = sorted(df["batsman"].dropna().unique().tolist())
all_teams   = sorted(CURRENT_SQUADS.keys())


# ═══════════════════════════════════════════════════════
# HERO HEADER  — much larger and dramatic
# ═══════════════════════════════════════════════════════

st.markdown("""
<div class="hero-outer">
    <div class="hero-glow"></div>
    <div class="hero-eyebrow">🏏 &nbsp; Powered by Machine Learning &nbsp; 🏏</div>
    <span class="hero-title">IPL MATCHUP</span><br>
    <span class="hero-title" style="background:linear-gradient(160deg,#fff 0%,rgba(255,255,255,0.4) 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        ENGINE
    </span>
    <p class="hero-sub">Find the perfect bowler for any batsman · any over · any pressure situation</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION 1 — PLAYERS & TEAMS
# ═══════════════════════════════════════════════════════

st.markdown('<p class="section-label">👥 Players & Teams</p>', unsafe_allow_html=True)
st.markdown('<p class="search-hint">💡 Type to search in any dropdown below</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    batsman = st.selectbox(
        "🧢 Batsman",
        options=all_batsmen,
        index=0,
        placeholder="Search or select a batsman...",
        help="Type a name to filter the list"
    )

with col2:
    batting_team = st.selectbox(
        "🏏 Batting Team",
        options=all_teams,
        placeholder="Select batting team...",
        help="The team currently batting"
    )

with col3:
    bowling_team_options = [t for t in all_teams if t != batting_team]
    bowling_team = st.selectbox(
        "🎳 Bowling Team",
        options=bowling_team_options,
        placeholder="Select bowling team...",
        help="Only bowlers from this team's current squad will be analysed"
    )

# Resolve bowlers from CURRENT_SQUADS (fixes stale CSV data issue)
available_bowlers = CURRENT_SQUADS.get(bowling_team, [])

if not available_bowlers:
    st.warning(f"No squad data found for **{bowling_team}**. Add them to CURRENT_SQUADS in app.py.")
    st.stop()

st.caption(f"📋 Analysing **{len(available_bowlers)} bowlers** from {bowling_team}'s current squad")


# ═══════════════════════════════════════════════════════
# SECTION 2 — MATCH SITUATION
# ═══════════════════════════════════════════════════════

st.markdown('<p class="section-label">📊 Match Situation</p>', unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    over = st.slider("📍 Current Over", 1, 20, 10)
with col5:
    wickets_fallen = st.slider("💥 Wickets Fallen", 0, 9, 2)

col6, col7, col8 = st.columns(3)
with col6:
    current_score = st.number_input("🏃 Current Score", min_value=0, max_value=400, value=80, step=1)
with col7:
    match_type = st.selectbox("🎯 Innings", ["1st Innings", "2nd Innings (Chase)"])
with col8:
    target = 0
    if "Chase" in match_type:
        target = st.number_input("🎯 Target", min_value=1, max_value=400, value=160, step=1)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Target only applies in a chase")


# ── PRESSURE INDEX ──
balls_bowled = max((over - 1) * 6, 1)
current_rr   = round((current_score / balls_bowled) * 6, 2)
is_chase     = "Chase" in match_type

if is_chase and target > 0:
    balls_left  = max((20 - over) * 6, 1)
    runs_needed = max(target - current_score, 0)
    req_rr      = round((runs_needed / balls_left) * 6, 2)
    rr_pressure = min(req_rr / max(current_rr, 1), 3.0)
else:
    req_rr      = 0.0
    rr_pressure = min(current_rr / 8.5, 2.0)

wicket_pressure = wickets_fallen * 0.4
pressure_index  = round(min(wicket_pressure + rr_pressure, 6.0), 2)
pct_pressure    = int((pressure_index / 6.0) * 100)

def phase_html(ov):
    if ov <= 6:   return '<span class="phase-chip phase-powerplay">⚡ Powerplay</span>'
    elif ov <= 15: return '<span class="phase-chip phase-middle">🔥 Middle</span>'
    else:          return '<span class="phase-chip phase-death">💀 Death</span>'

def p_color(pi):
    if pi < 2:   return "#64dc64"
    elif pi < 4: return "#f5a623"
    else:        return "#e8005a"

pcol = p_color(pressure_index)
req_display = f"{req_rr:.1f}" if is_chase else "—"
crr_color   = "green" if current_rr <= 8 else ("orange" if current_rr <= 12 else "red")
rrr_color   = "green" if req_rr <= 8 else ("orange" if req_rr <= 12 else "red")

st.markdown(f"""
<div class="ctx-card">
  <div class="ctx-grid">
    <div class="ctx-item">
      <div class="ctx-lbl">Phase</div>
      <div style="margin-top:4px">{phase_html(over)}</div>
    </div>
    <div class="ctx-item">
      <div class="ctx-lbl">Current RR</div>
      <div class="ctx-val {crr_color}">{current_rr}</div>
    </div>
    <div class="ctx-item">
      <div class="ctx-lbl">Required RR</div>
      <div class="ctx-val {rrr_color}">{req_display}</div>
    </div>
    <div class="ctx-item">
      <div class="ctx-lbl">Wickets Left</div>
      <div class="ctx-val {'red' if wickets_fallen>=7 else 'orange' if wickets_fallen>=5 else 'green'}">{10 - wickets_fallen}</div>
    </div>
  </div>
  <div>
    <div class="p-label" style="margin-top:16px;">Pressure Index</div>
    <div class="pressure-row">
      <div class="p-bar-bg">
        <div class="p-bar-fill" style="width:{pct_pressure}%;background:linear-gradient(90deg,#64dc64,{pcol});"></div>
      </div>
      <div class="p-val" style="color:{pcol};">{pressure_index}<span style="font-size:14px;opacity:.5;">/6</span></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# RUN ENGINE BUTTON
# ═══════════════════════════════════════════════════════

run_btn = st.button("🔍 ANALYSE MATCHUPS")

if run_btn:

    with st.spinner(f"Running batch analysis for {len(available_bowlers)} bowlers..."):
        # ── BATCHED PREDICTION — single transform+predict call ──
        matchup_df = batch_predict(
            bowlers=tuple(available_bowlers),   # tuple so st.cache_data can hash it
            batsman=batsman,
            batting_team=batting_team,
            bowling_team=bowling_team,
            over=over,
            pressure_index=pressure_index
        )

    best_df  = matchup_df.head(min(8, len(matchup_df))).reset_index(drop=True)
    worst_df = matchup_df.tail(min(5, len(matchup_df))).sort_values("Tactical Score").reset_index(drop=True)
    top      = best_df.iloc[0]

    # ── METRIC STRIP ──
    st.markdown(f"""
    <div class="metric-strip">
        <div class="metric-pill"><div class="val">{top['Dot %']}%</div><div class="lbl">Dot Ball</div></div>
        <div class="metric-pill"><div class="val">{top['Wicket %']}%</div><div class="lbl">Wicket Prob</div></div>
        <div class="metric-pill"><div class="val">{top['Boundary %']}%</div><div class="lbl">Boundary Risk</div></div>
        <div class="metric-pill"><div class="val">{top['Tactical Score']}</div><div class="lbl">Best Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"🏆 **Best pick:** `{top['Bowler']}` — top {bowling_team} bowler vs **{batsman}** | Over {over} | Pressure {pressure_index}/6")

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── BEST BOWLERS + BAR CHART ──
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<p class="section-label">🎯 Best Matchups</p>', unsafe_allow_html=True)
        max_score = best_df["Tactical Score"].max()
        for i, row in best_df.iterrows():
            pct_bar = int((row["Tactical Score"] / max_score) * 100) if max_score > 0 else 0
            medal   = ["🥇","🥈","🥉"][i] if i < 3 else str(i + 1)
            st.markdown(f"""
            <div class="bowler-row">
                <div class="rank-badge">{medal}</div>
                <div class="bowler-name">{row['Bowler']}</div>
                <div class="mini-bar-wrap">
                    <div class="mini-bar-bg">
                        <div class="mini-bar-fill" style="width:{pct_bar}%;background:linear-gradient(90deg,#f5a623,#ff6b35);"></div>
                    </div>
                </div>
                <div class="bowler-score">{row['Tactical Score']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-label">📊 Probability Breakdown</p>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Dot %",      x=best_df["Bowler"], y=best_df["Dot %"],      marker_color="#f5a623",              marker_line_width=0))
        fig_bar.add_trace(go.Bar(name="Wicket %",   x=best_df["Bowler"], y=best_df["Wicket %"],   marker_color="#e8005a",              marker_line_width=0))
        fig_bar.add_trace(go.Bar(name="Boundary %", x=best_df["Bowler"], y=best_df["Boundary %"], marker_color="rgba(255,255,255,.18)", marker_line_width=0))
        fig_bar.update_layout(
            barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="DM Sans"),
            legend=dict(orientation="h", y=-0.22),
            xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", ticksuffix="%"),
            margin=dict(l=0,r=0,t=10,b=0), height=370
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="fig_bar")

    # ── RADAR — TOP 3 ──
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">🕸 Top 3 Skill Radar</p>', unsafe_allow_html=True)

    categories   = ["Dot %", "Wicket %", "Four %", "Six %", "Boundary %"]
    colors_radar = [
        ("#f5a623", "rgba(245,166,35,0.12)"),
        ("#e8005a", "rgba(232,0,90,0.12)"),
        ("#00cfff", "rgba(0,207,255,0.12)"),
    ]
    fig_radar = go.Figure()
    for idx, row in best_df.head(3).iterrows():
        vals = [row[c] for c in categories] + [row[categories[0]]]
        lc, fc = colors_radar[idx]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["Bowler"],
            line_color=lc, fillcolor=fc
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.1)", color="rgba(255,255,255,.3)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="rgba(255,255,255,.5)")
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white", family="DM Sans"),
        legend=dict(orientation="h", y=-0.12),
        height=380, margin=dict(l=20,r=20,t=20,b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True, key="fig_radar")

    # ── BOWLERS TO AVOID ──
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">⚠️ Bowlers To Avoid</p>', unsafe_allow_html=True)

    col_w1, col_w2 = st.columns([1, 1])
    with col_w1:
        min_s = worst_df["Tactical Score"].min()
        max_s = worst_df["Tactical Score"].max()
        for i, row in worst_df.iterrows():
            pct_w = int(((row["Tactical Score"] - min_s) / (max_s - min_s + 1e-9)) * 100)
            st.markdown(f"""
            <div class="bowler-row" style="border-color:rgba(232,0,90,0.15);">
                <div class="rank-badge" style="color:#e8005a;">⚠</div>
                <div class="bowler-name">{row['Bowler']}</div>
                <div class="mini-bar-wrap">
                    <div class="mini-bar-bg">
                        <div class="mini-bar-fill" style="width:{pct_w}%;background:linear-gradient(90deg,#e8005a,#ff6b35);"></div>
                    </div>
                </div>
                <div class="bowler-score" style="color:#e8005a;">{row['Tactical Score']}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_w2:
        fig_w = go.Figure(go.Bar(
            x=worst_df["Bowler"], y=worst_df["Tactical Score"],
            marker=dict(color=worst_df["Tactical Score"],
                        colorscale=[[0,"#e8005a"],[1,"#ff6b35"]], showscale=False, line_width=0)
        ))
        fig_w.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="DM Sans"),
            xaxis=dict(tickangle=-30, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=0,r=0,t=10,b=0), height=300
        )
        st.plotly_chart(fig_w, use_container_width=True, key="fig_worst")

    # ── FULL TABLE ──
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    with st.expander(f"📋 Full Rankings — All {bowling_team} Bowlers"):
        st.dataframe(matchup_df.reset_index(drop=True), use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════

st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="footer">IPL TACTICAL MATCHUP ENGINE &nbsp;·&nbsp; ML-POWERED &nbsp;·&nbsp; BALL-BY-BALL INTELLIGENCE</p>',
    unsafe_allow_html=True
)