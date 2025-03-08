import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import shap
from calculate_weight import calculate_weight
from matplotlib import pyplot as plt
import xgboost as xgb

# function: global_recommendation(df: pd.DataFrame, y_col: str) -> np.ndarray
# Train a model using the dataframe. Analyze the model using SHAP values. Return the global recommendation based on the SHAP values.
def global_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = LogisticRegression()
    model.fit(X, y)

    # explain the model using SHAP
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the absolute mean shap_values for each feature
    shap_values = np.absolute(shap_values).mean(0)
    
    # Sort features based on the shap_values
    features = X.columns
    features = [x for _, x in sorted(zip(shap_values, features), reverse=True)]
    return features

# function: icare_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray
# Generate sample weights. Train a model using the dataframe and the sample weights. Analyze the model using SHAP values. Return the iCARE recommendation based on the SHAP values.
def icare_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str, return_weights: bool = False) -> np.ndarray:
    # calculate the weights
    weights = calculate_weight(df, sample, y_col)
    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = LogisticRegression()
    model.fit(X, y, sample_weight=weights)

    # explain the model using SHAP
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the absolute mean shap_values for each feature
    shap_values = np.absolute(shap_values).mean(0)
    
    # Sort features based on the shap_values
    features = X.columns
    features = [x for _, x in sorted(zip(shap_values, features), reverse=True)]
    if return_weights:
        return features, weights
    else:
        return features

# function: icare_cost_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str, cost_arr: np.ndarray) -> np.ndarray
# Generate sample weights. Train a model using the dataframe and the sample weights. Analyze the model using SHAP values. Return the iCARE recommendation based on the SHAP values divided by cost.
def icare_cost_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str, cost_arr) -> np.ndarray:
    # calculate the weights
    weights = calculate_weight(df, sample, y_col)
    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = LogisticRegression()
    model.fit(X, y, sample_weight=weights)

    # explain the model using SHAP
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the absolute mean shap_values for each feature
    shap_values = np.absolute(shap_values).mean(0)
    
    # Sort features based on the shap_values
    features = X.columns
    features = [x for _, x in sorted(zip(shap_values/(1+cost_arr), features), reverse=True)]
    return features

# function: eguided_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray
def eguided_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    # calculate the weights
    weights = calculate_weight(df, sample, y_col)

    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = LogisticRegression()
    model.fit(X, y)

    # weights contains the weights for each sample. We need to find the 100 closest samples to the sample
    X_copy = X.copy()
    # include weights to X
    X_copy["weights"] = weights
    # order the samples based on the weights
    X_copy = X_copy.sort_values(by="weights", ascending=False)
    # get the 100 closest samples
    X_copy = X_copy.head(100)
    # remove the weights column
    X_copy = X_copy.drop(columns=["weights"])

    # Create a new dataframe of imputed samples. For features known use the sample, for features unknown use the values from the 100 closest samples creating 100 imputed samples
    imputed_samples = pd.DataFrame()
    for index, row in X_copy.iterrows():
        imputed_sample = sample.copy()
        imputed_sample = imputed_sample.drop(columns=[y_col])
        # Check for feature in row but not in sample
        for feature in X_copy.columns:
            if feature not in sample.columns:
                imputed_sample[feature] = row[feature]
        imputed_samples = pd.concat([imputed_samples, imputed_sample], ignore_index=True)

    # explain the model using SHAP
    explainer = shap.LinearExplainer(model, imputed_samples)
    shap_values = explainer.shap_values(imputed_samples)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the variance shap_values for each feature
    shap_values = np.var(shap_values, axis=0)
    # # Plot it as a bar chart using matplotlib
    # plt.bar(X.columns, shap_values)
    # plt.show()

    # Sort features based on the shap_values
    features = imputed_samples.columns
    features = [x for _, x in sorted(zip(shap_values, features), reverse=True)]
    return features

# function: global_recommendation(df: pd.DataFrame, y_col: str) -> np.ndarray
# Train a model using the dataframe. Analyze the model using SHAP values. Return the global recommendation based on the SHAP values.
def global_recommendation_xgboost(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # explain the model using SHAP
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the absolute mean shap_values for each feature
    shap_values = np.absolute(shap_values).mean(0)
    
    # Sort features based on the shap_values
    features = X.columns
    features = [x for _, x in sorted(zip(shap_values, features), reverse=True)]
    return features

# function: icare_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray
# Generate sample weights. Train a model using the dataframe and the sample weights. Analyze the model using SHAP values. Return the iCARE recommendation based on the SHAP values.
def icare_recommendation_xgboost(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    # calculate the weights
    weights = calculate_weight(df, sample, y_col)
    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = xgb.XGBClassifier()
    model.fit(X, y, sample_weight=weights)

    # explain the model using SHAP
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the absolute mean shap_values for each feature
    shap_values = np.absolute(shap_values).mean(0)
    
    # Sort features based on the shap_values
    features = X.columns
    features = [x for _, x in sorted(zip(shap_values, features), reverse=True)]
    return features

# function: eguided_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray
def eguided_recommendation_xgboost(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    # calculate the weights
    weights = calculate_weight(df, sample, y_col)

    # train a logistic regression model
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # weights contains the weights for each sample. We need to find the 100 closest samples to the sample
    X_copy = X.copy()
    # include weights to X
    X_copy["weights"] = weights
    # order the samples based on the weights
    X_copy = X_copy.sort_values(by="weights", ascending=False)
    # get the 100 closest samples
    X_copy = X_copy.head(100)
    # remove the weights column
    X_copy = X_copy.drop(columns=["weights"])

    # Create a new dataframe of imputed samples. For features known use the sample, for features unknown use the values from the 100 closest samples creating 100 imputed samples
    imputed_samples = pd.DataFrame()
    for index, row in X_copy.iterrows():
        imputed_sample = sample.copy()
        imputed_sample = imputed_sample.drop(columns=[y_col])
        # Check for feature in row but not in sample
        for feature in X_copy.columns:
            if feature not in sample.columns:
                imputed_sample[feature] = row[feature]
        imputed_samples = pd.concat([imputed_samples, imputed_sample], ignore_index=True)

    # explain the model using SHAP
    explainer = shap.TreeExplainer(model, imputed_samples)
    shap_values = explainer.shap_values(imputed_samples)
    # shap.summary_plot(shap_values, X, plot_type="bar")
    # get the variance shap_values for each feature
    shap_values = np.var(shap_values, axis=0)
    # # Plot it as a bar chart using matplotlib
    # plt.bar(X.columns, shap_values)
    # plt.show()

    # Sort features based on the shap_values
    features = imputed_samples.columns
    features = [x for _, x in sorted(zip(shap_values, features), reverse=True)]
    return features

    


# df = pd.read_csv("Sample Dataset\pool_of_known_cases.csv")
# samples = pd.read_csv("Sample Dataset\samples.csv")
# y_col = "target"

# sample = pd.DataFrame(samples.iloc[25]).T
# sample['target'] = 0

# # print(eguided_recommendation(df, sample, y_col))
# print(icare_recommendation_xgboost(df, sample, y_col))
# print(eguided_recommendation_xgboost(df, sample, y_col))