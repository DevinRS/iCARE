import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from calculate_weight import calculate_weight
from create_case import create_case, create_case_split
from generate_recommendation import global_recommendation, icare_recommendation, icare_cost_recommendation, eguided_recommendation, global_recommendation_xgboost, icare_recommendation_xgboost, eguided_recommendation_xgboost, SFS_recommendation, LASSO_recommendation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

# # function: feature_oracle(df: pd.DataFrame, sample: pd.DataFrame, feature_list: list) -> pd.DataFrame
# # Given a complete dataframe and a sample from that dataframe, find the same sample in df and return the features in feature_list.
# def feature_oracle(df: pd.DataFrame, sample: pd.DataFrame, feature_list: list) -> pd.DataFrame:
#     # Find the corresponding sample in the dataframe
#     sample = df.iloc[sample.index]
#     # only include the features in feature_list
#     sample = sample[feature_list]
#     return sample

def feature_oracle(df: pd.DataFrame, sample: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    # Print debug information
    # print("Original DataFrame indices:", df.index)
    # print("Sample DataFrame indices:", sample.index)
    
    # Ensure the sample indices are valid
    valid_indices = sample.index.intersection(df.index)
    if len(valid_indices) != len(sample.index):
        raise IndexError("Some sample indices are out-of-bounds in the original DataFrame.")
    
    # Find the corresponding sample in the dataframe
    sample = df.loc[valid_indices]
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
        # icare_recommendation_list = icare_cost_recommendation(df_case, sample, y_col, cost_arr)
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

# function: run_analysis_lw(df: pd.DataFrame, y_col: str, number_of_features: int, str, static_features: list (optional), iteration: int (optional)) -> np.ndarray
# Create a case. Generate recommendation for that case. 
@ignore_warnings(category=ConvergenceWarning)
def run_analysis_lw(df: pd.DataFrame, y_col: str, number_of_features: int, static_features: list = None, iteration: int = 1) -> np.ndarray:
    global_prediction = []
    icare_prediction = []
    global_prediction_lw = []
    icare_prediction_lw = []
    y_actual = []
    df_original = df.copy()
    for i in range(iteration):
        print_statement = f"Iteration: {i+1}/{iteration} | "
        # Generate a case
        df_case, sample = create_case(df, number_of_features, y_col, static_features)

        # Generate recommendation
        global_recommendation_list = global_recommendation(df_case, sample, y_col)
        # icare_recommendation_list = icare_cost_recommendation(df_case, sample, y_col, cost_arr)
        icare_recommendation_list, weights = icare_recommendation(df_case, sample, y_col, return_weights=True)

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
        model_global_lw = LogisticRegression(max_iter=5000)
        model_icare_lw = LogisticRegression(max_iter=5000)
        model_global.fit(X_global, y)
        model_icare.fit(X_icare, y)
        model_global_lw.fit(X_global, y, sample_weight=weights)
        model_icare_lw.fit(X_icare, y, sample_weight=weights)

        # Predict the sample
        global_prediction.append(model_global.predict(global_sample.drop(y_col, axis=1))[0])
        icare_prediction.append(model_icare.predict(icare_sample.drop(y_col, axis=1))[0])
        global_prediction_lw.append(model_global_lw.predict(global_sample.drop(y_col, axis=1))[0])
        icare_prediction_lw.append(model_icare_lw.predict(icare_sample.drop(y_col, axis=1))[0])
        y_actual.append(sample[y_col].values[0])

        if i%(iteration/10) == 0:
            print(print_statement)

    return np.array(global_prediction), np.array(icare_prediction), np.array(global_prediction_lw), np.array(icare_prediction_lw), np.array(y_actual)

# function: run_analysis_lw_split(df: pd.DataFrame, y_col: str, number_of_features: int, str, static_features: list (optional), iteration: int (optional), split: float = 0.2) -> np.ndarray
# Create a validation split cases. Generate recommendation for that cases. 
@ignore_warnings(category=ConvergenceWarning)
def run_analysis_lw_split(df: pd.DataFrame, y_col: str, number_of_features: int, static_features: list = None, iteration: int = 1, split: float = 0.2) -> np.ndarray:
    global_prediction = []
    icare_prediction = []
    global_prediction_lw = []
    icare_prediction_lw = []
    y_actual = []
    df_original = df.copy()
    for i in range(iteration):
        print_statement = f"Iteration: {i+1}/{iteration} | "
        # Generate a case
        df_case, samples = create_case_split(df, number_of_features, y_col, static_features, split)

        # use pandas iterrows to iterate through the samples
        global_prediction_split = []
        icare_prediction_split = []
        global_prediction_lw_split = []
        icare_prediction_lw_split = []
        y_actual_split = []
        print_bool = True
        # add global marker so that we don't have to train global every time
        global_train = True
        global_features = None
        df_global = None
        model_global = None
        for index, sample in samples.iterrows():
            # convert to dataframe
            sample = pd.DataFrame(sample).T

            # Generate recommendation
            if global_train:
                global_recommendation_list = global_recommendation(df_case, sample, y_col)
            # icare_recommendation_list = icare_cost_recommendation(df_case, sample, y_col, cost_arr)
            icare_recommendation_list, weights = icare_recommendation(df_case, sample, y_col, return_weights=True)

            # Generate feature list
            if global_train:
                global_features = np.array(sample.columns)
                for feature in global_recommendation_list:
                    if feature not in global_features:
                        global_features = np.append(global_features, feature)
                        if print_bool:
                            print_statement += f"{feature} | "
                        break
            icare_features = np.array(sample.columns)
            for feature in icare_recommendation_list:
                if feature not in icare_features:
                    icare_features = np.append(icare_features, feature)
                    if print_bool:
                        print_statement += f"{feature} | "
                    break
            if print_bool:
                print_statement += f"{global_features} | "
                print_statement += f"{icare_features} | "
                print_bool = False

            # Generate sample
            global_sample = feature_oracle(df_original, sample, global_features)
            icare_sample = feature_oracle(df_original, sample, icare_features)

            # Train a model using df_case with either global_features or icare_features
            if global_train:
                df_global = df_case[global_features]
            df_icare = df_case[icare_features]
            X_global = df_global.drop(y_col, axis=1)
            X_icare = df_icare.drop(y_col, axis=1)
            y = df_case[y_col]
            if global_train:
                model_global = LogisticRegression(max_iter=5000)
                model_global.fit(X_global, y)
                global_train = False
            # model_global = LogisticRegression(max_iter=5000)
            model_icare = LogisticRegression(max_iter=5000)
            model_global_lw = LogisticRegression(max_iter=5000)
            model_icare_lw = LogisticRegression(max_iter=5000)
            # model_global.fit(X_global, y)
            model_icare.fit(X_icare, y)
            model_global_lw.fit(X_global, y, sample_weight=weights)
            model_icare_lw.fit(X_icare, y, sample_weight=weights)

            # Predict the sample
            global_prediction_split.append(model_global.predict_proba(global_sample.drop(y_col, axis=1))[:, 1][0])
            icare_prediction_split.append(model_icare.predict_proba(icare_sample.drop(y_col, axis=1))[:, 1][0])
            global_prediction_lw_split.append(model_global_lw.predict_proba(global_sample.drop(y_col, axis=1))[:, 1][0])
            icare_prediction_lw_split.append(model_icare_lw.predict_proba(icare_sample.drop(y_col, axis=1))[:, 1][0])
            y_actual_split.append(sample[y_col].values[0])

        global_prediction.append(global_prediction_split)
        icare_prediction.append(icare_prediction_split)
        global_prediction_lw.append(global_prediction_lw_split)
        icare_prediction_lw.append(icare_prediction_lw_split)
        y_actual.append(y_actual_split)

        if i%(iteration/10) == 0:
            print(print_statement)

    return np.array(global_prediction), np.array(icare_prediction), np.array(global_prediction_lw), np.array(icare_prediction_lw), np.array(y_actual)

# function: run_analysis_eguided(df: pd.DataFrame, y_col: str, number_of_features: int, str, static_features: list (optional), iteration: int (optional)) -> np.ndarray
# Create a case. Generate recommendation for that case. 
@ignore_warnings(category=ConvergenceWarning)
def run_analysis_eguided(df: pd.DataFrame, y_col: str, number_of_features: int, static_features: list = None, iteration: int = 1) -> np.ndarray:
    global_prediction = []
    icare_prediction = []
    eguided_prediction = []
    y_actual = []
    df_original = df.copy()
    for i in range(iteration):
        print_statement = f"Iteration: {i+1}/{iteration} | "
        # Generate a case
        df_case, sample = create_case(df, number_of_features, y_col, static_features)

        # Generate recommendation
        global_recommendation_list = global_recommendation(df_case, sample, y_col)
        # icare_recommendation_list = icare_cost_recommendation(df_case, sample, y_col, cost_arr)
        icare_recommendation_list = icare_recommendation(df_case, sample, y_col)
        eguided_recommendation_list = eguided_recommendation(df_case, sample, y_col)

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
        eguided_features = np.array(sample.columns)
        for feature in eguided_recommendation_list:
            if feature not in eguided_features:
                eguided_features = np.append(eguided_features, feature)
                print_statement += f"{feature} | "
                break
        print_statement += f"{global_features} | "
        print_statement += f"{icare_features} | "
        print_statement += f"{eguided_features} | "

        # Generate sample
        global_sample = feature_oracle(df_original, sample, global_features)
        icare_sample = feature_oracle(df_original, sample, icare_features)
        eguided_sample = feature_oracle(df_original, sample, eguided_features)

        # Train a model using df_case with either global_features or icare_features
        df_global = df_case[global_features]
        df_icare = df_case[icare_features]
        df_eguided = df_case[eguided_features]
        X_global = df_global.drop(y_col, axis=1)
        X_icare = df_icare.drop(y_col, axis=1)
        X_eguided = df_eguided.drop(y_col, axis=1)
        y = df_case[y_col]
        model_global = LogisticRegression(max_iter=5000)
        model_icare = LogisticRegression(max_iter=5000)
        model_eguided = LogisticRegression(max_iter=5000)
        model_global.fit(X_global, y)
        model_icare.fit(X_icare, y)
        model_eguided.fit(X_eguided, y)

        # Predict the sample
        global_prediction.append(model_global.predict(global_sample.drop(y_col, axis=1))[0])
        icare_prediction.append(model_icare.predict(icare_sample.drop(y_col, axis=1))[0])
        eguided_prediction.append(model_eguided.predict(eguided_sample.drop(y_col, axis=1))[0])
        y_actual.append(sample[y_col].values[0])

        if i%(iteration/10) == 0:
            print(print_statement)

    return np.array(global_prediction), np.array(icare_prediction), np.array(eguided_prediction), np.array(y_actual)


# function: run_analysis_all_split(df: pd.DataFrame, y_col: str, number_of_features: int, str, static_features: list (optional), iteration: int (optional), split: float = 0.2) -> np.ndarray
# Create a case. Generate recommendation for that case. 
@ignore_warnings(category=ConvergenceWarning)
def run_analysis_all_split(df: pd.DataFrame, y_col: str, number_of_features: int, static_features: list = None, iteration: int = 1, split: float = 0.2) -> np.ndarray:
    global_prediction = []
    sfs_prediction = []
    lasso_prediction = []
    icare_prediction = []
    eguided_prediction = []
    y_actual = []
    df_original = df.copy()
    for i in range(iteration):
        print_statement = f"Iteration: {i+1}/{iteration} | "
        # Generate a case
        df_case, samples = create_case_split(df, number_of_features, y_col, static_features, split)

        # use pandas iterrows to iterate through the samples
        global_prediction_split = []
        sfs_prediction_split = []
        lasso_prediction_split = []
        icare_prediction_split = []
        eguided_prediction_split = []
        y_actual_split = []
        print_bool = True
        # add global marker so that we don't have to train global every time
        global_train = True
        global_features = None
        df_global = None
        model_global = None
        # add sfs marker so that we don't have to train sfs every time
        sfs_train = True
        sfs_features = None
        df_sfs = None
        model_sfs = None
        # add lasso marker so that we don't have to train lasso every time
        lasso_train = True
        lasso_features = None
        df_lasso = None
        model_lasso = None
        for index, sample in samples.iterrows():
            # convert to dataframe
            sample = pd.DataFrame(sample).T

            # Generate recommendation
            if global_train:
                global_recommendation_list = global_recommendation(df_case, sample, y_col)
            if sfs_train:
                sfs_recommendation_list = SFS_recommendation(df_case, sample, y_col)
            if lasso_train:
                lasso_recommendation_list = LASSO_recommendation(df_case, sample, y_col)
            # icare_recommendation_list = icare_cost_recommendation(df_case, sample, y_col, cost_arr)
            icare_recommendation_list = icare_recommendation(df_case, sample, y_col)
            eguided_recommendation_list = eguided_recommendation(df_case, sample, y_col)

            # Generate feature list
            if global_train:
                global_features = np.array(sample.columns)
                for feature in global_recommendation_list:
                    if feature not in global_features:
                        global_features = np.append(global_features, feature)
                        if print_bool:
                            print_statement += f"{feature} | "
                        break
            if sfs_train:
                sfs_features = np.array(sample.columns)
                for feature in sfs_recommendation_list:
                    if feature not in sfs_features:
                        sfs_features = np.append(sfs_features, feature)
                        if print_bool:
                            print_statement += f"{feature} | "
                        break
            if lasso_train:
                lasso_features = np.array(sample.columns)
                for feature in lasso_recommendation_list:
                    if feature not in lasso_features:
                        lasso_features = np.append(lasso_features, feature)
                        if print_bool:
                            print_statement += f"{feature} | "
                        break
            icare_features = np.array(sample.columns)
            for feature in icare_recommendation_list:
                if feature not in icare_features:
                    icare_features = np.append(icare_features, feature)
                    if print_bool:
                        print_statement += f"{feature} | "
                    break
            eguided_features = np.array(sample.columns)
            for feature in eguided_recommendation_list:
                if feature not in eguided_features:
                    eguided_features = np.append(eguided_features, feature)
                    if print_bool:
                        print_statement += f"{feature} | "
                    break
            if print_bool:
                print_statement += f"{global_features} | "
                print_statement += f"{sfs_features} | "
                print_statement += f"{lasso_features} | "
                print_statement += f"{icare_features} | "
                print_statement += f"{eguided_features} | "
                print_bool = False

            # Generate sample
            global_sample = feature_oracle(df_original, sample, global_features)
            sfs_sample = feature_oracle(df_original, sample, sfs_features)
            lasso_sample = feature_oracle(df_original, sample, lasso_features)
            icare_sample = feature_oracle(df_original, sample, icare_features)
            eguided_sample = feature_oracle(df_original, sample, eguided_features)

            # Train a model using df_case with either global_features or icare_features
            if global_train:
                df_global = df_case[global_features]
            if sfs_train:
                df_sfs = df_case[sfs_features]
            if lasso_train:
                df_lasso = df_case[lasso_features]
            df_icare = df_case[icare_features]
            df_eguided = df_case[eguided_features]
            X_global = df_global.drop(y_col, axis=1)
            X_sfs = df_sfs.drop(y_col, axis=1)
            X_lasso = df_lasso.drop(y_col, axis=1)
            X_icare = df_icare.drop(y_col, axis=1)
            X_eguided = df_eguided.drop(y_col, axis=1)
            y = df_case[y_col]
            if global_train:
                model_global = LogisticRegression(max_iter=5000)
                model_global.fit(X_global, y)
                global_train = False
            if sfs_train:
                model_sfs = LogisticRegression(max_iter=5000)
                model_sfs.fit(X_sfs, y)
                sfs_train = False
            if lasso_train:
                model_lasso = LogisticRegression(max_iter=5000)
                model_lasso.fit(X_lasso, y)
                lasso_train = False
            # model_global = LogisticRegression(max_iter=5000)
            model_icare = LogisticRegression(max_iter=5000)
            model_eguided = LogisticRegression(max_iter=5000)
            # model_global.fit(X_global, y)
            model_icare.fit(X_icare, y)
            model_eguided.fit(X_eguided, y)

            # Predict the sample
            global_prediction_split.append(model_global.predict(global_sample.drop(y_col, axis=1))[0])
            sfs_prediction_split.append(model_sfs.predict(sfs_sample.drop(y_col, axis=1))[0])
            lasso_prediction_split.append(model_lasso.predict(lasso_sample.drop(y_col, axis=1))[0])
            icare_prediction_split.append(model_icare.predict(icare_sample.drop(y_col, axis=1))[0])
            eguided_prediction_split.append(model_eguided.predict(eguided_sample.drop(y_col, axis=1))[0])
            y_actual_split.append(sample[y_col].values[0])

        global_prediction.append(global_prediction_split)
        sfs_prediction.append(sfs_prediction_split)
        lasso_prediction.append(lasso_prediction_split)
        icare_prediction.append(icare_prediction_split)
        eguided_prediction.append(eguided_prediction_split)
        y_actual.append(y_actual_split)

        if i%(iteration/10) == 0:
            print(print_statement)

    return np.array(global_prediction), np.array(sfs_prediction), np.array(lasso_prediction), np.array(icare_prediction), np.array(eguided_prediction), np.array(y_actual)

# function: run_analysis_xgboost(df: pd.DataFrame, y_col: str, number_of_features: int, str, static_features: list (optional), iteration: int (optional)) -> np.ndarray
# Create a case. Generate recommendation for that case. 
@ignore_warnings(category=ConvergenceWarning)
def run_analysis_xgboost(df: pd.DataFrame, y_col: str, number_of_features: int, static_features: list = None, iteration: int = 1, cost_arr: list = None) -> np.ndarray:
    global_prediction = []
    icare_prediction = []
    eguided_prediction = []
    y_actual = []
    df_original = df.copy()
    for i in range(iteration):
        print_statement = f"Iteration: {i+1}/{iteration} | "
        # Generate a case
        df_case, sample = create_case(df, number_of_features, y_col, static_features)

        # Generate recommendation
        global_recommendation_list = global_recommendation_xgboost(df_case, sample, y_col)
        # icare_recommendation_list = icare_cost_recommendation(df_case, sample, y_col, cost_arr)
        icare_recommendation_list = icare_recommendation_xgboost(df_case, sample, y_col)
        eguided_recommendation_list = eguided_recommendation_xgboost(df_case, sample, y_col)

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
        eguided_features = np.array(sample.columns)
        for feature in eguided_recommendation_list:
            if feature not in eguided_features:
                eguided_features = np.append(eguided_features, feature)
                print_statement += f"{feature} | "
                break
        print_statement += f"{global_features} | "
        print_statement += f"{icare_features} | "
        print_statement += f"{eguided_features} | "

        # Generate sample
        global_sample = feature_oracle(df_original, sample, global_features)
        icare_sample = feature_oracle(df_original, sample, icare_features)
        eguided_sample = feature_oracle(df_original, sample, eguided_features)

        # Train a model using df_case with either global_features or icare_features
        df_global = df_case[global_features]
        df_icare = df_case[icare_features]
        df_eguided = df_case[eguided_features]
        X_global = df_global.drop(y_col, axis=1)
        X_icare = df_icare.drop(y_col, axis=1)
        X_eguided = df_eguided.drop(y_col, axis=1)
        y = df_case[y_col]
        model_global = xgb.XGBClassifier()
        model_icare = xgb.XGBClassifier()
        model_eguided = xgb.XGBClassifier()
        model_global.fit(X_global, y)
        model_icare.fit(X_icare, y)
        model_eguided.fit(X_eguided, y)

        # Predict the sample
        global_prediction.append(model_global.predict(global_sample.drop(y_col, axis=1))[0])
        icare_prediction.append(model_icare.predict(icare_sample.drop(y_col, axis=1))[0])
        eguided_prediction.append(model_eguided.predict(eguided_sample.drop(y_col, axis=1))[0])
        y_actual.append(sample[y_col].values[0])

        if i%(iteration/10) == 0:
            print(print_statement)

    return np.array(global_prediction), np.array(icare_prediction), np.array(eguided_prediction), np.array(y_actual)

