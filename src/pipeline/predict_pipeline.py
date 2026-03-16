import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object



class CustomData:

    def __init__(
        self,
        batsman: str,
        bowler: str,
        batting_team: str,
        bowling_team: str,
        over: int,
        ball: int
    ):
        self.batsman = batsman
        self.bowler = bowler
        self.batting_team = batting_team
        self.bowling_team = bowling_team
        self.over = over
        self.ball = ball


    def get_data_as_dataframe(self):

        try:

            input_dict = {
                "batsman": [self.batsman],
                "bowler": [self.bowler],
                "batting_team": [self.batting_team],
                "bowling_team": [self.bowling_team],
                "over": [self.over],
                "ball": [self.ball]
            }

            return pd.DataFrame(input_dict)

        except Exception as e:
            raise CustomException(e, sys)



class PredictPipeline:

    def __init__(self):
        pass


    def predict_probabilities(self, features):

        try:

            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)

            probabilities = model.predict_proba(data_scaled)

            classes = model.classes_

            prob_df = pd.DataFrame(probabilities, columns=classes)

            return prob_df

        except Exception as e:
            raise CustomException(e, sys)




    def tactical_score(self, prob_df):

        try:

            dot_prob = prob_df.get("dot", 0)
            wicket_prob = prob_df.get("wicket", 0)

            boundary_prob = (
                prob_df.get("four", 0)
                + prob_df.get("six", 0)
            )

            score = (0.6 * dot_prob) + (1.0 * wicket_prob) - (0.7 * boundary_prob)

            return score.values[0]

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":

    data = CustomData(
        batsman="V Kohli",
        bowler="R Ashwin",
        batting_team="Royal Challengers Bangalore",
        bowling_team="Chennai Super Kings",
        over=10,
        ball=3
    )

    df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()

    probs = pipeline.predict_probabilities(df)

    score = pipeline.tactical_score(probs)

    print("Outcome Probabilities:")
    print(probs)

    print("\nTactical Score:", score)