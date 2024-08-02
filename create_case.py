import pandas as pd
import random

# function: create_case(df: pd.DataFrame, number_of_features: int, y_col: str, static_features: list (optional))
# Takes a dataframe and generate a random single sample with the defined number of features. Features are selected randomly. If static_features is provided, they will be included in the sample, and the rest of the features will be selected randomly. Force the number of static_features to be less than number_of_features.
def create_case(df: pd.DataFrame, number_of_features: int, y_col: str, static_features: list = None) -> pd.DataFrame:
    if static_features is not None:
        assert len(static_features) <= number_of_features, "Number of static features must be less than or equal to number of features"
    else:
        static_features = []
    # available_columns is the list of columns that is not in the static_features
    available_columns = [col for col in df.columns if col not in static_features]
    # remove y_col from the available_columns
    available_columns = [col for col in available_columns if col != y_col]
    features = static_features + random.sample(available_columns, number_of_features - len(static_features)) + [y_col]
    # pick a random row from the dataframe
    sample = df.sample()
    # select only the features selected
    sample = sample[features]
    # remove the sample from the dataframe
    df = df.drop(sample.index)
    return df, sample