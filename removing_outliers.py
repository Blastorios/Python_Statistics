###Full credits to @author: ddo033

###(ALREADY WORKING)
def quantiles_per_column_per_group(df, col, group_col, cut_off = 1.5):
    
    ##df = the DataFrame you want to scan through
    ##col = the column with the float/int values to do the pd.quantile over
    ##group_col = the column which holds the groups that you want to divide col's values from
    ##cut_ff = the StdDev range you wish to apply to the dataset (generally '3' is regarded as an outlier)
    
    dfc = df.copy()
    for group in dfc[group_col].unique():
        data_per_group = dfc[col][dfc[group_col] == group]
        
        Q1 = data_per_group.quantile(0.25)
        Q3 = data_per_group.quantile(0.75)
        IQR = Q3 - Q1
        
        lowOutlier = Q1 - cut_off * IQR
        
        highOutlier = Q3 + cut_off * IQR
        
        for values in data_per_group:
            if values >= lowOutlier:
                if values <= highOutlier:
                    pass
                else:
                    dfc.loc[dfc[col] == values, col] = np.nan
            else:
                dfc.loc[dfc[col] == values, col] = np.nan
    
    return dfc


###(NOT WORKING YET)
def scrape_data_for_plot(df, col, group_col, cut_off = 3):

    ##df = the DataFrame you want to scan through
    ##col = the column with the float/int values to do the pd.quantile over
    ##group_col = the column which holds the groups that you want to divide col's values from
    ##cut_ff = the StdDev range you wish to apply to the dataset (generally '3' is regarded as an outlier)
    
    dfs_to_plot = []
    for group in df[group_col].unique():
        temp_df = pd.DataFrame()
        data_per_group = df[col][df[group_col] == group]

        mean = data_per_group.median()
        std = data_per_group.std()
        cut_off_high = mean + (cut_off*std)
        cut_off_low = mean - (cut_off*std)
        
        data_per_group = data_per_group[data_per_group < cut_off_high]
        data_per_group = data_per_group[data_per_group > cut_off_low]
        temp_df[col] = data_per_group
        temp_df[group_col] = group
        dfs_to_plot.append(temp_df)
    return pd.concat(dfs_to_plot)
