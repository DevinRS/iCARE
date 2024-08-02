import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from calculate_weight import calculate_weight
from create_case import create_case
from generate_recommendation import global_recommendation, icare_recommendation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# function: feature_oracle(df: pd.DataFrame, sample: pd.DataFrame, feature_list: list) -> pd.DataFrame
# Given a complete dataframe and a sample from that dataframe, find the same sample in df and return the features in feature_list.
def feature_oracle(df: pd.DataFrame, sample: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    # Find the corresponding sample in the dataframe
    sample = df.iloc[sample.index]
    # only include the features in feature_list
    sample = sample[feature_list]
    return sample

# function: run_analysis(df: pd.DataFrame, y_col: str, number_of_features: int, str, static_features: list (optional), iteration: int (optional)) -> np.ndarray
# Create a case. Generate recommendation for that case. 
@ignore_warnings(category=ConvergenceWarning)
def run_analysis(df: pd.DataFrame, y_col: str, number_of_features: int, static_features: list = None, iteration: int = 1) -> np.ndarray:
    global_prediction = []
    icare_prediction = []
    y_actual = []
    df_original = df.copy()
    for i in range(iteration):
        print_statement = f"Iteration: {i+1}/{iteration} | "
        # Generate a case
        df_case, sample = create_case(df, number_of_features, y_col, static_features)

        # Generate recommendation
        global_recommendation_list = global_recommendation(df_case, sample, y_col)
        icare_recommendation_list = icare_recommendation(df_case, sample, y_col)

        # Generate feature list
        global_features = np.array(sample.columns)
        for feature in global_recommendation_list:
            if feature not in global_features:
                global_features = np.append(global_features, feature)
                print_statement += f"{feature} | "
                break
        icare_features = np.array(sample.columns)
        for feature in icare_recommendation_list:
            if feature not in icare_features:
                icare_features = np.append(icare_features, feature)
                print_statement += f"{feature} | "
                break
        print_statement += f"{global_features} | "
        print_statement += f"{icare_features} | "

        # Generate sample
        global_sample = feature_oracle(df_original, sample, global_features)
        icare_sample = feature_oracle(df_original, sample, icare_features)

        # Train a model using df_case with either global_features or icare_features
        df_global = df_case[global_features]
        df_icare = df_case[icare_features]
        X_global = df_global.drop(y_col, axis=1)
        X_icare = df_icare.drop(y_col, axis=1)
        y = df_case[y_col]
        model_global = LogisticRegression(max_iter=5000)
        model_icare = LogisticRegression(max_iter=5000)
        model_global.fit(X_global, y)
        model_icare.fit(X_icare, y)

        # Predict the sample
        global_prediction.append(model_global.predict(global_sample.drop(y_col, axis=1))[0])
        icare_prediction.append(model_icare.predict(icare_sample.drop(y_col, axis=1))[0])
        y_actual.append(sample[y_col].values[0])

        if i%(iteration/10) == 0:
            print(print_statement)

    return np.array(global_prediction), np.array(icare_prediction), np.array(y_actual)

