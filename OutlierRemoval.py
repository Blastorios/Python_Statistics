#Start with importing all required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Set the path where the dataframe is located
path = 'path/to/file.csv'

df = pd.read_csv(path, delimiter = '\t')
dfc = df.copy()

#Columns to group by:
groupcol = "drugs"
valuecol = "speed030_mean"

#Excluding 'Dead' animals
dfc = dfc[dfc['displacement_max'] >= 500]

#Create plotting functions to lower the amount of written code
def set_axis(the_plot, Title, t_font, X_Axis, x_font, Y_Axis, y_font):
    the_plot.set_title(Title, fontsize=int(t_font))
    the_plot.set_xlabel(X_Axis, fontsize=int(x_font))
    the_plot.set_ylabel(Y_Axis, fontsize=int(y_font))

def x_labels(m, labels, r, h, f):
    m.set_xticklabels(labels, rotation=r, ha=h, fontsize=f)

#Plotting a matplotlib native boxplot to look for outliers as they are natively calculated.
#This will already give an idea if the code is required or not
to_plot = []
plot_labels = []
genes = dfc[groupcol].unique()

for gene in genes:
    to_plot.append(dfc[valuecol][dfc[groupcol] == gene].values)
    plot_labels.append(gene)

fig, ax = plt.subplots(1,2)
plt.subplots_adjust(left=10,right=11)

rawbox = sns.boxplot(data=to_plot[0:2990], ax=ax[0])
set_axis(rawbox, 'None-Statistical Plotting', 15, groupcol, 12, valuecol, 12)
x_labels(rawbox, plot_labels, 40, "right", 12)

rawbox2 = sns.boxplot(data=to_plot[0:2990], showfliers=False, ax=ax[1])
set_axis(rawbox2, 'Statistical Plotting', 15, groupcol, 12, valuecol, 12)
x_labels(rawbox2, plot_labels, 40, "right", 12)

#Applying the Pandas native quantile calculator to determine the outliers of every individual attribute per group.
#Followed by checking the numbers to their respective groups
def quantiles_per_column_per_group(df, col, group_col, cut_off = 1.5):
    dfc = df.copy()
    for group in dfc[group_col].unique():
        data_per_group = dfc[col][dfc[group_col] == group]
        
        Q1 = data_per_group.quantile(0.25)
        Q3 = data_per_group.quantile(0.75)
        IQR = Q3 - Q1
        
        lowOutlier = Q1 - cut_off * IQR
        
        highOutlier = Q3 + cut_off * IQR
        
        #Currently, the following for loop is slow but worked well
        for values in data_per_group:
            if values >= lowOutlier:
                if values <= highOutlier:
                    pass
                else:
                    dfc.loc[(dfc[col] == values) & (dfc[group_col] == group), col] = np.nan
            else:
                dfc.loc[(dfc[col] == values) & (dfc[group_col] == group), col] = np.nan
        #Specifying both the column value and group column value to make sure we dont delete a non-outlier\
        #from a different group which simply resembles the outlier from the calculated group.
    
    return dfc[col]

#Applying the outlier removal function to the dataset and assigning it to a new dataframe.
dfclean = dfc.copy()
for col in dfc.columns:
    if dfc[col].dtype == np.float64:
        dfclean[col] = quantiles_per_column_per_group(dfc, col, "drugs", cut_off=1.5);

#Plotting the new dataframe to compare. The new plots with 'Showfliers'=False should now be identical
#to the ones we plotted earlier if you were to look at it without the outliers
genel=[]
for genen in dfc[groupcol].unique():
    genel.append(genen)

Yaxn = valuecol + " Value Count"
fig, ax = plt.subplots(1,3)
plt.subplots_adjust(left=15,right=17)

fig1 = sns.boxplot(data=dfc, showfliers=True, x = groupcol, y = valuecol, ax=ax[0]);
x_labels(fig1, genel, 40, "right", 12)
set_axis(fig1, 'Original DataFrame', 15, 'Chemogenetic State', 12, Yaxn, 12)

fig2 = sns.boxplot(data=dfc, showfliers=False, x = groupcol, y = valuecol, ax=ax[1]);
x_labels(fig2, genel, 40, "right", 12)
set_axis(fig2, 'Original DataFrame - No Outliers', 15, 'Chemogenetic State', 12, Yaxn, 12)

fig3 = sns.boxplot(data=dfclean, showfliers=True, x = groupcol, y = valuecol, ax=ax[2]);
x_labels(fig3, genel, 40, "right", 12)
set_axis(fig3, 'Cleaned DataFrame', 15, 'Chemogenetic State', 12, Yaxn, 12)

#If all plots are as they should be (equal) save the new dataframe to a new destination
path = 'path/to/store/new/dataframe.csv'

dfclean.to_csv(path, sep = "\t")
