import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

RAW_CSV_PATH = "data/processed/clean_ipl_data.csv"

class CustomData:

    def __init__(
        self,
        batsman: str,
        bowler: str,
        batting_team: str,
        bowling_team: str,
        over: int,
        pressure_index: float = 1.0   
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
                "ball":         [self.pressure_index],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:

    def __init__(self):
        self.model        = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")
        self.data = pd.read_csv(RAW_CSV_PATH)
        self._resolve_economy_col()

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
            raise ValueError("Could not find bowler_economy column.")

    def normalize_name(self, name: str) -> str:
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
        
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[0][0]} {parts[-1]}"
        return name

    def get_team_bowlers(self, bowling_team: str) -> list:
        df = self.data
        if "bowling_team" not in df.columns:
            for alt in ["fielding_team", "field_team", "team_bowling"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "bowling_team"})
                    self.data = df
                    break

        team_df = df[df["bowling_team"] == bowling_team]
        if len(team_df) == 0:
            return sorted(df["bowler"].dropna().unique().tolist())
        return sorted(team_df["bowler"].dropna().unique().tolist())

    @staticmethod
    def get_phase(over: int) -> str:
        if over <= 6:
            return "powerplay"
        elif over <= 15:
            return "middle"
        else:
            return "death"

    def get_stats(self, batsman: str, bowler: str) -> dict:
        df = self.data
        
        batsman_norm = self.normalize_name(batsman)
        bowler_norm = self.normalize_name(bowler)

        row = df[(df["batsman"] == batsman_norm) & (df["bowler"] == bowler_norm)]

        if len(row) == 0:
            # FIX: Pull individual player stats instead of global means
            bat_df = df[df["batsman"] == batsman_norm]
            bowl_df = df[df["bowler"] == bowler_norm]

            bat_sr = bat_df["batsman_strike_rate"].iloc[0] if len(bat_df) > 0 else df["batsman_strike_rate"].mean()
            bowl_econ = bowl_df["bowler_economy"].iloc[0] if len(bowl_df) > 0 else df["bowler_economy"].mean()
            bowl_dr = bowl_df["dismissal_rate"].mean() if len(bowl_df) > 0 else df["dismissal_rate"].mean()

            return {
                "strike_rate_vs_bowler": bat_sr,
                "dismissal_rate":        bowl_dr,
                "bowler_economy":        bowl_econ,
                "batsman_strike_rate":   bat_sr,
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