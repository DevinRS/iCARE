import pandas as pd
import numpy as np

# function: euclidean_distance(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray
# Takes two samples and calculate the euclidean distance between them. The function returns the euclidean distance. Only calculate the same features in both samples and ignore the y_col.
def euclidean_distance(df: pd.DataFrame, sample: pd.DataFrame, y_col: str) -> np.ndarray:
    # get the common columns
    common_columns = sample.columns
    # remove y_col from the common columns
    common_columns = [col for col in common_columns if col != y_col]
    df = df[common_columns]
    sample = sample[common_columns]
    # calculate the euclidean distance for all rows
    distance = []
    for i in range(len(df)):
        x1 = df.iloc[i].values
        x2 = sample.values
        distance.append(np.sqrt(np.sum((x1 - x2)**2)))
    return np.array(distance)

def distance_to_weight(distance):
    return 1/(distance+1e-9)

# function: calculate_weight(df: pd.DataFrame, single_case: pd.DataFrame) -> np.ndarray
# Takes a dataframe and a single sample and calculate the weight between each sample in the dataframe and the single sample. The weight is calculated using the inverse of the euclidean distance between the samples. The function returns a numpy array with the weights.
def calculate_weight(df: pd.DataFrame, single_case: pd.DataFrame, y_col: str) -> np.ndarray:
    distances = euclidean_distance(df, single_case, y_col)
    weights = distance_to_weight(distances)
    return weights

