import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    # ---------------------------------------------------
    # PREPROCESSOR OBJECT
    # ---------------------------------------------------

    def get_obj(self):

        try:

            numerical_cols = [
                "over",
                "ball",
                "bowler_economy",
                "batsman_strike_rate",
                "strike_rate_vs_bowler",
                "dismissal_rate",
                "avg_runs"
            ]

            categorical_cols = [
                "batsman",
                "bowler",
                "batting_team",
                "bowling_team",
                "venue",
                "match_phase"
            ]


            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )


            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )


            logging.info("Creating preprocessing pipelines")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    # ---------------------------------------------------
    # DATA TRANSFORMATION
    # ---------------------------------------------------

    def data_trans(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded")

            df = pd.concat([train_df, test_df])


            # ---------------------------------------------------
            # FIX DUPLICATE ECONOMY COLUMNS
            # ---------------------------------------------------

            if "bowler_economy_x" in df.columns:
                df["bowler_economy"] = df["bowler_economy_x"]

            if "bowler_economy_y" in df.columns:
                df["bowler_economy"] = df["bowler_economy_y"]

            cols_to_drop = [c for c in ["bowler_economy_x","bowler_economy_y"] if c in df.columns]

            df.drop(columns=cols_to_drop, inplace=True)


            # ---------------------------------------------------
            # MATCH PHASE FEATURE
            # ---------------------------------------------------

            def get_phase(over):

                if over <= 6:
                    return "powerplay"
                elif over <= 15:
                    return "middle"
                else:
                    return "death"

            df["match_phase"] = df["over"].apply(get_phase)


            # ---------------------------------------------------
            # CREATE OUTCOME TARGET
            # ---------------------------------------------------

            df["outcome"] = "dot"

            df.loc[df["batsman_runs"] == 1, "outcome"] = "single"
            df.loc[df["batsman_runs"] == 2, "outcome"] = "double"
            df.loc[df["batsman_runs"] == 3, "outcome"] = "triple"
            df.loc[df["batsman_runs"] == 4, "outcome"] = "four"
            df.loc[df["batsman_runs"] == 6, "outcome"] = "six"

            if "player_dismissed" in df.columns:
                df.loc[df["player_dismissed"].notna(), "outcome"] = "wicket"


            # ---------------------------------------------------
            # SPLIT BACK TRAIN TEST
            # ---------------------------------------------------

            train_df = df.iloc[:len(train_df)]
            test_df = df.iloc[len(train_df):]


            target_column = "outcome"


            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]


            # ---------------------------------------------------
            # APPLY PREPROCESSOR
            # ---------------------------------------------------

            preprocessor = self.get_obj()

            logging.info("Applying preprocessing object")

            input_feature_tr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_te = preprocessor.transform(input_feature_test_df)


            # Convert sparse matrices
            if hasattr(input_feature_tr, "toarray"):
                input_feature_tr = input_feature_tr.toarray()

            if hasattr(input_feature_te, "toarray"):
                input_feature_te = input_feature_te.toarray()


            target_feature_train_df = target_feature_train_df.values.reshape(-1,1)
            target_feature_test_df = target_feature_test_df.values.reshape(-1,1)


            train_arr = np.hstack((input_feature_tr, target_feature_train_df))
            test_arr = np.hstack((input_feature_te, target_feature_test_df))


            # ---------------------------------------------------
            # SAVE PREPROCESSOR
            # ---------------------------------------------------

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)