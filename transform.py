import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntProgress
from typing import *

def import_data(path: str, method='hdf') -> pd.DataFrame:
    """
    Import Function for Pandas Dataframes through hdf or csv.

    Parameters:
    path = path to datafile
    method = Method by which Pandas has to read the data files
            Options = 'hdf' / 'csv'
    """
    try:
        import pandas as pd
    except Exception as e:
        print(f"{e}, Install the module with 'pip install <module>'")

    if method == 'hdf':
        dataframe = pd.read_hdf(path)

    if method == 'csv':
        dataframe = pd.read_csv(path, sep='\t') #Set seperator so the Dataframe has the correct layout

    return dataframe

def get_features(df: pd.DataFrame, feature_mark: str, banned: list) -> list:
    """
    Get the dataframe features as a list by scanning for a repetitive marking within the
    feature's terminology
    """
    try:
        import pandas as pd
    except Exception as e:
        print(f"{e}, Install the module with 'pip install <module>'")

    features = [x for x in df.columns if feature_mark in x and df[x].dtype != object and x not in banned]

    return features


def skewness_values(df: pd.DataFrame, value_columns: list) -> dict:
    """
    Skewness Function to calculate skewness of a given dataset
    """
    try:
        import scipy.stats as scs
    except Exception as e:
        print(f"{e}, Install the module with 'pip install <module>'")
    skewness_values = {}

    for value_column in value_columns:
        stat, pval = scs.skewtest(df[value_column])
        skewness_values[value_column] = pval

    return skewness_values


def transform_data(df: pd.DataFrame, value_column: str, method='pos'):
    """
    Transformation Function to transform an entire dataset based on a given Transformation method.

    Parameters:

    df = A Pandas dataframe that contains the values to transform
    value_columns = The dataframe columns to be affected
    method = The transformation method by which the values will be transformed, depending on the skewness of a data distribution. (skewness > 0 = pos, skewness < 0 = neg)
            Options = 'pos' / 'neg'
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        print(f"{e}, Install the module with 'pip install <module>'")

    if method == 'neg':
        df[value_column] = df[value_column].transform(lambda x: 0.5*np.log((1+x)/(1-x)))
        print(f"Data {value_column} Transformed from a positive skew to center")
        return df
    
    elif method == 'pos':
        df[value_column] = df[value_column].transform(lambda x: 0.5*np.log(x))
        print(f"Data {value_column} Transformed from a negative skew to center")
        return df

def selective_transformation(path, feature_mark, banned):
    df = import_data(path)
    features = get_features(df, feature_mark, banned)
    skewness_vals = skewness_values(df, features)
    f = widgets.IntProgress(min=0, max=len(features)-1)
    display(f)
    for f.value, key in enumerate(skewness_vals):
        try:
            if skewness_vals[key] > 0:
                df = transform_data(df, key)
            elif skewness_vals[key] < 0:
                df = transform_data(df, key, method='neg')
            else:
                pass
        except Exception as e:
            print(f"Received Incorrect Input, {e}")
    return df
    

data = selective_transformation(example_df_path, feature_mark_to_select_by, features_list_with_names_that_shouldnt_be_included)
