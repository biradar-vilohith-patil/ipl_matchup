import sys
import difflib
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

RAW_CSV_PATH = "data/processed/clean_ipl_data.csv"

# -------------------------------------------------------
# CUSTOM DATA (user input)
# -------------------------------------------------------
class CustomData:
    def __init__(
        self, batsman: str, bowler: str, batting_team: str, 
        bowling_team: str, over: int, pressure_index: float = 1.0
    ):
        self.batsman        = batsman
        self.bowler         = bowler
        self.batting_team   = batting_team
        self.bowling_team   = bowling_team
        self.over           = over
        self.pressure_index = pressure_index

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame({
                "batsman":      [self.batsman],
                "bowler":       [self.bowler],
                "batting_team": [self.batting_team],
                "bowling_team": [self.bowling_team],
                "over":         [self.over],
                "ball":         [self.pressure_index],
            })
        except Exception as e:
            raise CustomException(e, sys)


# -------------------------------------------------------
# PREDICT PIPELINE
# -------------------------------------------------------
class PredictPipeline:
    def __init__(self):
        self.model        = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        self.data         = pd.read_csv(RAW_CSV_PATH)
        self._resolve_economy_col()
        
        # Cache all unique real names from the dataset for Fuzzy Matching
        batsmen = self.data["batsman"].dropna().unique().tolist()
        bowlers = self.data["bowler"].dropna().unique().tolist()
        self.unique_players = list(set(batsmen + bowlers))

    def _resolve_economy_col(self):
        df = self.data
        if "bowler_economy" in df.columns:
            self._economy_col = "bowler_economy"
        elif "bowler_economy_x" in df.columns:
            df["bowler_economy"] = df["bowler_economy_x"].combine_first(df.get("bowler_economy_y", df["bowler_economy_x"]))
            self._economy_col = "bowler_economy"
        elif "bowler_economy_y" in df.columns:
            df["bowler_economy"] = df["bowler_economy_y"]
            self._economy_col = "bowler_economy"
        else:
            raise ValueError("Could not find bowler_economy column.")

    # --------------------------------------------------
    # INTELLIGENT NAME NORMALIZATION
    # --------------------------------------------------
    def normalize_name(self, name: str) -> str:
        # 1. Check strict known aliases first
        aliases = {
            "Virat Kohli": "V Kohli", "Rohit Sharma": "RG Sharma", "Jasprit Bumrah": "JJ Bumrah",
            "Hardik Pandya": "HH Pandya", "Suryakumar Yadav": "SA Yadav", "MS Dhoni": "MS Dhoni",
            "KL Rahul": "KL Rahul", "Shubman Gill": "Shubman Gill", "Rashid Khan": "Rashid Khan",
            "Ravindra Jadeja": "RA Jadeja", "Glenn Maxwell": "GJ Maxwell", "Rishabh Pant": "RR Pant",
            "Faf du Plessis": "F du Plessis", "David Warner": "DA Warner", "Sunil Narine": "SP Narine",
            "Andre Russell": "AD Russell", "Trent Boult": "TA Boult", "Yuzvendra Chahal": "YS Chahal"
        }
        if name in aliases: 
            return aliases[name]

        # 2. Exact match check
        if name in self.unique_players: 
            return name

        # 3. Standard First Initial + Last Name check
        parts = name.split()
        if len(parts) >= 2:
            short_name = f"{parts[0][0]} {parts[-1]}"
            if short_name in self.unique_players: 
                return short_name

        # 4. THE FIX: Fuzzy String Matching against actual dataset names
        # This catches "Piyush Chawla" -> "PP Chawla" smoothly
        matches = difflib.get_close_matches(name, self.unique_players, n=1, cutoff=0.5)
        if matches: 
            return matches[0]

        # 5. Last resort: Last name subset match
        if len(parts) >= 1:
            last_name_matches = [p for p in self.unique_players if parts[-1] in p]
            if last_name_matches: 
                return last_name_matches[0]

        return name

    # --------------------------------------------------
    # HISTORICAL STATS W/ SYNTHETIC BLENDING
    # --------------------------------------------------
    def get_stats(self, batsman: str, bowler: str) -> dict:
        df = self.data

        batsman_norm = self.normalize_name(batsman)
        bowler_norm  = self.normalize_name(bowler)

        row = df[(df["batsman"] == batsman_norm) & (df["bowler"] == bowler_norm)]

        if len(row) == 0:
            # LOGICAL FIX: We must build unique features for unplayed matchups
            bat_df = df[df["batsman"] == batsman_norm]
            bowl_df = df[df["bowler"] == bowler_norm]

            # Get their individual career stats
            bat_sr    = bat_df["batsman_strike_rate"].iloc[0] if len(bat_df) > 0 else 130.0
            bowl_econ = bowl_df["bowler_economy"].iloc[0] if len(bowl_df) > 0 else 8.5
            
            # Avoid calculating mean on empty series
            if len(bowl_df) > 0 and "dismissal_rate" in bowl_df.columns and not bowl_df["dismissal_rate"].isna().all():
                bowl_dr = bowl_df["dismissal_rate"].mean()
            else:
                bowl_dr = 0.05

            # THE MATH FIX: Synthetic Feature Blending
            # If they haven't faced each other, blend the Batsman's SR with the Bowler's Economy.
            # An economy of 8.5 equates to an expected strike rate of ~141 (8.5 * 100 / 6).
            expected_bowler_sr = (bowl_econ * 100) / 6
            synthetic_sr_vs_bowler = (bat_sr + expected_bowler_sr) / 2

            return {
                "strike_rate_vs_bowler": synthetic_sr_vs_bowler, # Unique for EVERY bowler now
                "dismissal_rate":        bowl_dr,
                "bowler_economy":        bowl_econ,
                "batsman_strike_rate":   bat_sr,
                "avg_runs":              df["avg_runs"].mean() if not df.empty else 1.5,
                "venue":                 df["venue"].mode()[0] if not df.empty else "Unknown"
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

    @staticmethod
    def get_phase(over: int) -> str:
        if over <= 6: return "powerplay"
        elif over <= 15: return "middle"
        else: return "death"

    def get_team_bowlers(self, bowling_team: str) -> list:
        # Kept for compatibility if used elsewhere
        pass 

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    def predict_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        pass # Using batch_predict in app.py instead