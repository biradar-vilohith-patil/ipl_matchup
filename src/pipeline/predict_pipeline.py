import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

# Consistent CSV filename — must match data_ingestion.py
RAW_CSV_PATH = "data/processed/clean_ipl_data.csv"


# -------------------------------------------------------
# CUSTOM DATA (user input)
# -------------------------------------------------------

class CustomData:

    def __init__(
        self,
        batsman: str,
        bowler: str,
        batting_team: str,
        bowling_team: str,
        over: int,
        pressure_index: float = 1.0   # replaces raw "ball" — computed in app.py
    ):
        self.batsman        = batsman
        self.bowler         = bowler
        self.batting_team   = batting_team
        self.bowling_team   = bowling_team
        self.over           = over
        self.pressure_index = pressure_index

    def get_data_as_dataframe(self):
        try:
            data = {
                "batsman":      [self.batsman],
                "bowler":       [self.bowler],
                "batting_team": [self.batting_team],
                "bowling_team": [self.bowling_team],
                "over":         [self.over],
                # pressure_index maps into the "ball" column the preprocessor expects
                "ball":         [self.pressure_index],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------
# PREDICT PIPELINE
# -------------------------------------------------------

class PredictPipeline:

    def __init__(self):
        self.model        = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")

        # Load historical data — same filename as data_ingestion.py
        self.data = pd.read_csv(RAW_CSV_PATH)

        # Resolve bowler_economy column name once at init
        self._resolve_economy_col()

    # --------------------------------------------------
    # RESOLVE ECONOMY COLUMN NAME
    # --------------------------------------------------

    def _resolve_economy_col(self):
        df = self.data

        if "bowler_economy" in df.columns:
            self._economy_col = "bowler_economy"

        elif "bowler_economy_x" in df.columns:
            if "bowler_economy_y" in df.columns:
                df["bowler_economy"] = df["bowler_economy_x"].combine_first(df["bowler_economy_y"])
            else:
                df["bowler_economy"] = df["bowler_economy_x"]
            self._economy_col = "bowler_economy"

        elif "bowler_economy_y" in df.columns:
            df["bowler_economy"] = df["bowler_economy_y"]
            self._economy_col = "bowler_economy"

        else:
            raise ValueError(
                "Could not find bowler_economy column. "
                "Expected: bowler_economy, bowler_economy_x, or bowler_economy_y"
            )

    # --------------------------------------------------
    # GET BOWLERS FOR A SPECIFIC TEAM  ← KEY FIX
    # Only returns bowlers who have actually bowled for
    # the selected bowling team in the historical dataset.
    # --------------------------------------------------

    def get_team_bowlers(self, bowling_team: str) -> list:
        df = self.data

        # Handle alternate column names gracefully
        if "bowling_team" not in df.columns:
            for alt in ["fielding_team", "field_team", "team_bowling"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "bowling_team"})
                    self.data = df
                    break

        team_df = df[df["bowling_team"] == bowling_team]

        if len(team_df) == 0:
            # Team not found — graceful fallback to all bowlers
            return sorted(df["bowler"].dropna().unique().tolist())

        return sorted(team_df["bowler"].dropna().unique().tolist())

    # --------------------------------------------------
    # MATCH PHASE
    # --------------------------------------------------

    @staticmethod
    def get_phase(over: int) -> str:
        if over <= 6:
            return "powerplay"
        elif over <= 15:
            return "middle"
        else:
            return "death"

    # --------------------------------------------------
    # FETCH HISTORICAL FEATURES
    # --------------------------------------------------

    def get_stats(self, batsman: str, bowler: str) -> dict:
        df = self.data

        row = df[
            (df["batsman"] == batsman) &
            (df["bowler"]  == bowler)
        ]

        if len(row) == 0:
            return {
                "strike_rate_vs_bowler": df["strike_rate_vs_bowler"].mean(),
                "dismissal_rate":        df["dismissal_rate"].mean(),
                "bowler_economy":        df["bowler_economy"].mean(),
                "batsman_strike_rate":   df["batsman_strike_rate"].mean(),
                "avg_runs":              df["avg_runs"].mean(),
                "venue":                 df["venue"].mode()[0]
            }

        row = row.iloc[0]

        return {
            "strike_rate_vs_bowler": row["strike_rate_vs_bowler"],
            "dismissal_rate":        row["dismissal_rate"],
            "bowler_economy":        row["bowler_economy"],
            "batsman_strike_rate":   row["batsman_strike_rate"],
            "avg_runs":              row["avg_runs"],
            "venue":                 row["venue"]
        }

    # --------------------------------------------------
    # PREDICT PROBABILITIES
    # --------------------------------------------------

    def predict_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            batsman = features["batsman"].values[0]
            bowler  = features["bowler"].values[0]
            over    = features["over"].values[0]

            stats = self.get_stats(batsman, bowler)

            features = features.copy()
            features["match_phase"]           = self.get_phase(over)
            features["strike_rate_vs_bowler"] = stats["strike_rate_vs_bowler"]
            features["dismissal_rate"]        = stats["dismissal_rate"]
            features["bowler_economy"]        = stats["bowler_economy"]
            features["batsman_strike_rate"]   = stats["batsman_strike_rate"]
            features["avg_runs"]              = stats["avg_runs"]
            features["venue"]                 = stats["venue"]

            data_scaled = self.preprocessor.transform(features)

            probs   = self.model.predict_proba(data_scaled)
            prob_df = pd.DataFrame(probs, columns=self.model.classes_)

            return prob_df

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    # TACTICAL SCORE
    # Positive → bowler-favourable | Negative → batsman-favourable
    # --------------------------------------------------

    def tactical_score(self, prob_df: pd.DataFrame) -> float:
        def _get_prob(outcome: str) -> float:
            return float(prob_df[outcome].values[0]) if outcome in prob_df.columns else 0.0

        dot_prob      = _get_prob("dot")
        wicket_prob   = _get_prob("wicket")
        boundary_prob = _get_prob("four") + _get_prob("six")

        score = (0.6 * dot_prob) + (1.0 * wicket_prob) - (0.7 * boundary_prob)

        return round(score, 4)