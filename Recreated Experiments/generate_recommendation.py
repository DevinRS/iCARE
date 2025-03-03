import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import shap
from calculate_weight import calculate_weight

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
def icare_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
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
    return features, weights