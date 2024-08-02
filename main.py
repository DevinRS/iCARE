import pandas as pd
import numpy as np
from calculate_weight import calculate_weight
from create_case import create_case
from generate_recommendation import global_recommendation, icare_recommendation
from run_analysis import run_analysis
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Function: generate_recommendation(df1: pd.DataFrame, df2: pd.DataFrame, y_col: str) -> pd.DataFrame
# Given two dataframes, generate recommendations for samples in df2. Append the recommendations to df2 as a new column.
@ignore_warnings(category=ConvergenceWarning)
def generate_recommendation(df1: pd.DataFrame, df2: pd.DataFrame, y_col: str) -> pd.DataFrame:
    print("Generating recommendations for samples...")
    # To make sure df2 works properly, if df2 doesn't have y_col, add y_col to df2 with all zeros
    delete_y_col = False
    if y_col not in df2.columns:
        df2[y_col] = 0
        delete_y_col = True

    # Get the columns in df2
    features_in_samples = df2.columns

    # Iterate through the samples in df2
    for index, sample in df2.iterrows():
        # Generate recommendation for the sample, make sure to pass in sample as a DataFrame
        global_recommendation_list = global_recommendation(df1, pd.DataFrame(sample).T, y_col)
        icare_recommendation_list = icare_recommendation(df1, pd.DataFrame(sample).T, y_col)

        # Append the recommendations to the sample
        for feature in global_recommendation_list:
            if feature not in features_in_samples:
                # Append the feature to a new column named "global_recommendation"
                df2.at[index, "global_recommendation"] = feature
                break
        for feature in icare_recommendation_list:
            if feature not in features_in_samples:
                # Append the feature to a new column named "icare_recommendation"
                df2.at[index, "icare_recommendation"] = feature
                break

    # If y_col is not in df2, delete y_col
    if delete_y_col:
        df2.drop(columns=[y_col], inplace=True)

    return df2

# Function: analyze_for_sample(df1: pd.DataFrame, df2: pd.DataFrame, y_col: str, iteration: int) -> pd.DataFrame
# Given two dataframes, generate performance metrics comparing global and icare recommendations.
@ignore_warnings(category=ConvergenceWarning)
def analyze_for_sample(df1: pd.DataFrame, df2: pd.DataFrame, y_col: str, iteration: int = 100) -> pd.DataFrame:
    print("Analyzing performance metrics for samples...")
    # get the columns in df2
    static_features = [col for col in df2.columns if col != y_col]
    # get the number of features
    number_of_features = len(static_features)
    global_prediction, icare_prediction, y_actual = run_analysis(df1, y_col, number_of_features, static_features, iteration)

    # print the performance metrics
    print("Performance Metrics:")
    print("Global Recommendation:")
    print(classification_report(y_actual, global_prediction))
    print("Icare Recommendation:")
    print(classification_report(y_actual, icare_prediction))

# Function: analyze_for_num_feature(df1: pd.DataFrame, y_col: str, number_of_feature: int, static_features: list, iteration: int) -> pd.DataFrame
# Given a dataframe and number_of_feature, generate performance metrics comparing global and icare recommendations.
@ignore_warnings(category=ConvergenceWarning)
def analyze_for_num_feature(df1: pd.DataFrame, y_col: str, number_of_feature: int, static_features: list = None, iteration: int = 100) -> pd.DataFrame:
    print("Analyzing performance metrics for number of features...")
    global_prediction, icare_prediction, y_actual = run_analysis(df1, y_col, number_of_feature, static_features=static_features, iteration=iteration)

    # print the performance metrics
    print("Performance Metrics:")
    print("Global Recommendation:")
    print(classification_report(y_actual, global_prediction))
    print("Icare Recommendation:")
    print(classification_report(y_actual, icare_prediction))

# Process input from the user
# main.py generate [df1 file path] [df2 file path] [y_col] [output file path]
# main.py analyze_sample [df1 file path] [df2 file path] [y_col] [iteration]
# main.py analyze_feature [df1 file path] [y_col] [number_of_feature] [static_features] [iteration]
import sys
if len(sys.argv) < 2:
    print("Please provide the correct arguments")
else:
    if sys.argv[1] == "generate":
        try:
            df1 = pd.read_csv(sys.argv[2])
            df2 = pd.read_csv(sys.argv[3])
            y_col = sys.argv[4]
            output_file = sys.argv[5]
            df2 = generate_recommendation(df1, df2, y_col)
            df2.to_csv(output_file, index=False)
        except:
            print("Please provide arguments in the following format: main.py generate [df1 file path] [df2 file path] [y_col] [output file path]")
    elif sys.argv[1] == "analyze_sample":
        try:
            df1 = pd.read_csv(sys.argv[2])
            df2 = pd.read_csv(sys.argv[3])
            y_col = sys.argv[4]
            iteration = int(sys.argv[5])
            analyze_for_sample(df1, df2, y_col, iteration)
        except:
            print("Please provide arguments in the following format: main.py analyze_sample [df1 file path] [df2 file path] [y_col] [iteration]")
    elif sys.argv[1] == "analyze_feature":
        try:
            df1 = pd.read_csv(sys.argv[2])
            y_col = sys.argv[3]
            number_of_feature = int(sys.argv[4])
            static_features = sys.argv[5].split(",")
            iteration = int(sys.argv[6])
            analyze_for_num_feature(df1, y_col, number_of_feature, static_features, iteration)
        except:
            print("Please provide arguments in the following format: main.py analyze_feature [df1 file path] [y_col] [number_of_feature] [static_features] [iteration]")
    else:
        print("Please provide the correct arguments")
