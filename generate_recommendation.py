import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import shap
from calculate_weight import calculate_weight
from matplotlib import pyplot as plt
import xgboost as xgb
from create_case import create_case
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV

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

def SFS_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    random_key = random.randint(0, 1000000)
    y = df[y_col]
    # train a logistic regression model
    column_not_in_sample = [col for col in df.columns if col not in sample.columns]

    performance = []
    for feature in column_not_in_sample:
        column_include = sample.columns.tolist() + [feature]
        X = df[column_include]
        X = X.drop(y_col, axis=1)
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_key)
        # train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        # get the score
        score = accuracy_score(y_test, model.predict(X_test))
        performance.append((feature, score))
    # sort the performance
    performance = sorted(performance, key=lambda x: x[1], reverse=True)
    # get the features
    features = [x[0] for x in performance]
    # append the rest of the feature that is in df but not in features
    features = features + [col for col in df.columns if col not in features]
    # drop y_col
    features = [col for col in features if col != y_col]
    return features

def LASSO_recommendation(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    random_key = random.randint(0, 1000000)
    y = df[y_col]
    X = df.drop(columns=[y_col])  # Exclude target variable

    # Train Lasso Logistic Regression (L1 penalty)
    model = LogisticRegressionCV(
        penalty='l1', solver='liblinear', cv=5, random_state=random_key
    ).fit(X, y)

    # Get absolute coefficients
    feature_importance = abs(model.coef_[0])

    # Rank features based on coefficient magnitude
    feature_ranking = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)
    
    # Extract sorted feature names
    sorted_features = [feature for feature, coef in feature_ranking]

    return sorted_features