import pandas as pd
import numpy as np
from scipy import stats
from typing import *
from traceback import print_stack

def quantiles_per_column_per_group(df: pd.DataFrame, col: str, group_col: str, cut_off = 1.5) -> pd.DataFrame:
    """
    A Function to extract the outliers per group of a specific value column. Simply address the dataframe
    to be affected and the column that contains the groups. It will return a dataframe in which the outliers
    have been set to np.nan.
    """

    dfc = df.copy() #Safe Practice

    for group in dfc[group_col].unique():
        try:
            data_per_group = dfc[col][dfc[group_col] == group] #Get the values out of the column, per group
        except Exception as exc:
            print(f"Could not find the groups, received the following error:\n
                    {exc}"
            print_stack(exc)
        #Calculate the Quantile params
        Q1 = data_per_group.quantile(0.25)
        Q3 = data_per_group.quantile(0.75)
        IQR = Q3 - Q1

        lowOutlier = Q1 - cut_off * IQR

        highOutlier = Q3 + cut_off * IQR

        try:
            for values in data_per_group: #Apply the quantiles to every value within the dataframe
                if values >= lowOutlier:
                    if values <= highOutlier:
                        pass
                    else:
                        dfc.loc[dfc[col] == values, col] = np.nan
                else:
                    dfc.loc[dfc[col] == values, col] = np.nan
        except Exception as exc:
            print(f"Could not apply the quantiles, received the following error:\n
                    {exc}")
            print_stack(exc)      

    return dfc #Returning the copied dataframe
