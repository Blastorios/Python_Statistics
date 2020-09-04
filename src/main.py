from typing import *
from scipy import stats
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PythonStatistics(object):
    """
    Python Statistics (PS) is a generalistic statistic applier for students. Easy in use, it is limited to a handfull of functions:
        1. QQ-Plots
        2. Shapiro-Wilk
        3. Student's T-test
        4. MannWhitney-U test

    Returning, if desired, ready-to-use plots for reports.

    The script is jupyter-friendly and can run in any editor with a link to python >= 3.

    For more info, check out https://www.blastorios.dev/pythonstatistics or https://www.github.com/Blastorios/python_statistics  
    """
    def __init__(self, path=None, Mode=None):
        super().__init__()
        self.path = path
        self.Mode = Mode

    def get_dataframe (self, path: string, command="") -> pd.DataFrame:
        """
        Extract the data from .csv, .pickle or .hdf files. Returning a pandas dataframe format to use in future plotting.
        Input:
            - path: (str) the relative path from the workdir to where the data file is located
            - command: (str) [optional] if the file extension is not listed inside of the
        
        Output:
            - something
        
        """
        if self.command != "":
            exec(str(self.command))

        try:
            fileName, fileExtension = os.path.splitext(self.path)
        except Exception as e:
            return f"Encountered Error {e}, Please ensure the filename does not contain any other '.' except for the file extension."

        if fileExtension.lower() == '.csv':
            df = pd.read_csv(self.path)
        elif fileExtension.lower() == '.pickle':
            df = pd.read_pickle(self.path)
        elif fileExtension.lower() == '.hdf':
            df = pd.read_hdf(self.path)
        else: raise TypeError("Could not find an appropiate file extension. \
        Use either csv, pickle or hfd with the base command. Use a custom command for any other extension")

        if df.empty(): raise ValueError(f"DataFrame {self.path} has no values")

        if df.isna():
            print(f"Summarizing missing values:\n{df.isna().sum().sum()}\n{df.isna().sum()}\n")
            na_input = input("Proceed with empty values? (y/n): ")
            if na_input == 'y':
                return df
            else:
                column_or_row_removal = input("Delete the associated row or column of data? (r/c): ")
                if column_or_row_removal == 'r':
                    pass

                elif column_or_row_removal == 'c':
                    pass

                else:
                    print("Please use 'r' or 'c'.")

    def transform_data (self, dataframe: pd.DataFrame, value_columns: list = None, method='default') -> pd.DataFrame:
        """
        Tranform the columns of your respective dataframe.
        Input:
        Output:
        """
        try:
            import numpy as np
            import pandas as pd
        except Exception as e:
            return f"Error {e}, please install numpy and pandas using 'pip install numpy pandas'."

        def _skewness_value(self, df: pd.DataFrame, value_columns: list) -> dict:
            try: 
                import scipy.stats as scs
            except Exception as e:
                return f"Could not import Scipy Statistics. Encountered the following Error: {e}"
            skewness_values = {}
            for value_col in value_columns:
                stat, pval = scs.skewtest(df[value_col])
                skewness_values[value_column] = pval

        if (method == 'default') and (value_columns is not None):
            skew_values = _skewness_value(dataframe, value_columns)

        if value_columns is None:
            for value_col in value_columns:
                if method == 'logp':
                    for col in dataframe.columns:
                        if col.dtype == 'float64':
                            dataframe[value_column] = dataframe[value_col].transform(lambda x: 0.5*np.log(x))
                    return dataframe

                elif method == 'logn':
                    for col in dataframe.columns:
                        if col.dtype == 'float64':
                            dataframe[value_column] = dataframe[value_col].transform(lambda x: 0.5*np.log((1+x)/(1-x)))
                    return dataframe

                elif method == 'repr':
                    for col in dataframe.columns:
                        if col.dtype == 'float64':
                            dataframe[value_column] = dataframe[value_col].transform(lambda x: 1/x)
                    return dataframe

                else:
                    return f"Method {method} is not an option. Please select either logp, logn or repr."
        else:
            pass
            

    def remove_outliers(self, dataframe: pd.DataFrame, 
    dataframe_group_column: str,
    skip_columns: list,
    cut_off=3) -> pd.DataFrame:

        """
        Remove Outliers by specifiying the dataframe column that holds the group. Automatically iterating over every column except those that have been specified to be skipped.
        """

        try:
            import pandas as pd
        except Exception as e:
            return f"Error {e}, please install pandas with 'pip install pandas'."

        dfc = dataframe.copy()

        for group in dfc[dataframe_group_column].unique():
            for col in dfc:
                if col not in skip_columns and dfc[col].dtypes == 'float64':
                    try:
                        data_per_group = dfc[col][dfc[dataframe_group_column] == group]
                    except Exception as e:
                        return f"Experienced Error: {e}. Could not extract data from '{col}' based on '{group}'."
                    try:    
                        Q1 = data_per_group.quantile(0.25)
                        Q3 = data_per_group.quantile(0.75)
                        IQR = Q3 - Q1
                    except Exception as e:
                        return f"Experienced Error: {e}. Could not extract the quantiles from the selected data '{col}' in group '{group}'."
                    
                    lowOutlier = Q1 - cut_off * IQR
                    highOutlier = Q3 + cut_off * IQR

                    #Currently, the following for loop is slow but worked well
                    for values in data_per_group:
                        if values >= lowOutlier:
                            if values <= highOutlier:
                                pass
                            else:
                                dfc.loc[(dfc[col] == values) & (dfc[dataframe_group_column] == group), col] = np.nan
                        else:
                            dfc.loc[(dfc[col] == values) & (dfc[dataframe_group_column] == group), col] = np.nan
            #Specifying both the column value and group column value to make sure we dont delete a non-outlier\
            #from a different group which simply resembles the outlier from the calculated group.
        
        return dfc

    def make_qqplots(self, title: str,
    ylabel: str,
    xlabel: str,
    bins = 10,
    figsize = [10,10],
    mutliple_plots = False,
    ylabel2 = '',
    xlabel2 = '',
    bins2 = '',
    figsize2 = [10,10],
    tight_layout = False
    ) -> plt.subplots :

        """
        Plot up to 2 QQ-plot variants of your selected data.
        """

        return None
        
