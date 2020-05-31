#Start with importing all required modules
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import violinplot, boxplot
import numpy as np
from statistics import stdev

#Import the dataframes
df1 = pd.read_csv('data_holding_file_1.csv', delimiter='\t')
df2 = pd.read_csv('data_holding_file_2.csv', delimiter='\t')

#Define the columns that we will group by
df1_groups_col = 'Stage'
df2_groups_col = 'Chemogenetic_State'

#Set the different required parameters
df1_stages = df1.Stage.unique()
df2_stages = df2.Chemogenetic_State.unique()
df1_stages = sorted(df1_stages)
df2_stages = sorted(df2_stages)
features = [x for x in df1.columns if x.startswith('_pf_') and df1[x].dtype != object]
features = features[:-3]

#Assign all individual columns to a single nested dict, divided either by stage or chemogenetic state
df1_vector = {}
df2_vector = {}

for feature in features:
  df1_vector[feature] = {}
  for stage in df1_stages:
      df1_vector[feature][stage] = df1[feature][df1[df1_groups_col] == stage].values
        
for feature in features:
  df2_vector[feature] = {}
  for stage in df2_stages:
      df2_vector[feature][stage] = df2[feature][df2[df2_groups_col] == stage].values

#Plot all required values from the data vectors
for feature in features:
  print(f"{feature}:")
  for stage in df1_stages:
    data = df1_vector[feature][stage]
    print(f"STAGE:{stage} MEAN = {np.mean(data):.3f} / STD.DEV = {stdev(data):.4f}")
  print("\n")

for feature in features:
  print(f"{feature}:")
  for stage in df2_stages:
    data = df2_vector[feature][stage]
    print(f"STATE:{stage} MEAN = {np.mean(data):.3f} / STD.DEV = {stdev(data):.4f}")
  print("\n")

#Define plotting parameters
letters = ['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I']

ylabels = {}
labels = ['ΔÛ(l)', 'ΔÛ(r)', 'ΔÛ(m)', 'ΔÛ(z)', 'Û^2(z)', 'Û^2(min)', 'Fi(l)/s', 'Fi(r)/s', 'seconds']
for ii, feature in enumerate(features):
    ylabels[feature] = labels[ii]

#Plot all data by putting a matplotlib boxplot over a matplotlib violinplot
flierprops = dict(marker='.', markerfacecolor='black', markersize=5,
                  linestyle='none')

fig, axs = plt.subplots(3,3, figsize = (14,12))

fig.subplots_adjust(left=0.25, wspace=0.3, hspace=0.3)

for key,ax,letter in zip(df1_vector, axs.reshape(-1), letters):
    ax.violinplot(df1_vector[key].values(), showextrema=False)
    b = ax.boxplot(df1_vector[key].values(), notch= True, flierprops=flierprops, showfliers=True)
    ax.set_xticks(range(1,len(df1_stages)+1))
    ax.set_xticklabels(df1_stages)
    ax.set_xlabel('Developmental Stages')
    ax.set_ylabel(ylabels[key])
    ax.set_title(f"{letter} - {key}")
    fig.tight_layout()

#Instantly plot the required raw data values which are created by matplotlib    
    n_per_stage = df1.groupby('Stage').count()
    
    counts = n_per_stage['ImgUUID']
    m22 = b['medians'][0].get_ydata()
    m23 = b['medians'][1].get_ydata()
    m24 = b['medians'][2].get_ydata()
    m25 = b['medians'][3].get_ydata()
    m26 = b['medians'][4].get_ydata()
    s22 = b['whiskers'][0].get_ydata() 
    e22 = b['whiskers'][1].get_ydata()
    s23 = b['whiskers'][2].get_ydata() 
    e23 = b['whiskers'][3].get_ydata()
    s24 = b['whiskers'][4].get_ydata()
    e24 = b['whiskers'][5].get_ydata()
    s25 = b['whiskers'][6].get_ydata()
    e25 = b['whiskers'][7].get_ydata()
    s26 = b['whiskers'][8].get_ydata()
    e26 = b['whiskers'][9].get_ydata()
    
    print(f"{key}-VALUES:\n STAGE 22: BOTTOM {s22[1]:.4f} / MEDIAN {m22[0]:.4f} / TOPPER {e22[1]:.4f}\nSTAGE 23: BOTTOM {s23[1]:.4f} / MEDIAN {m23[0]:.4f} / TOPPER {e23[1]:.4f}\nSTAGE 24: BOTTOM {s24[1]:.4f} / MEDIAN {m24[0]:.4f} / TOPPER {e24[1]:.4f}\nSTAGE 25: BOTTOM {s25[1]:.4f} / MEDIAN {m25[0]:.4f} / TOPPER {e25[1]:.4f}\nSTAGE 26: BOTTOM {s26[1]:.4f} / MEDIAN {m26[0]:.4f} / TOPPER {e26[1]:.4f}\n")
# fig.savefig('filename.png')

#Repeat the plotting process for the second dataframe
flierprops = dict(marker='.', markerfacecolor='black', markersize=5,
                  linestyle='none')

fig, axs = plt.subplots(3,3, figsize = (14,12))

fig.subplots_adjust(left=0.25, wspace=0.3, hspace=0.3)

for key,ax,letter in zip(df2_vector, axs.reshape(-1), letters):
  ax.violinplot(df2_vector[key].values(), showextrema=False)
  b = ax.boxplot(df2_vector[key].values(), notch= True, flierprops=flierprops, showfliers=True)
  ax.set_xticks(range(1,len(df2_stages)+1))
  ax.set_xticklabels(df2_stages)
  ax.set_xlabel('Chemogenetic States')
  ax.set_ylabel(ylabels[key])
  ax.set_title(f"{letter} - {key}")
  fig.tight_layout()
  
  n_per_stage = df2.groupby('Chemogenetic_State').count()
  
  counts = n_per_stage['ImgUUID']
  m3 = b['medians'][0].get_ydata()
  m3c = b['medians'][1].get_ydata()
  m4 = b['medians'][2].get_ydata()
  m4c = b['medians'][3].get_ydata()
  s3 = b['whiskers'][0].get_ydata() 
  e3 = b['whiskers'][1].get_ydata()
  s3c = b['whiskers'][2].get_ydata() 
  e3c = b['whiskers'][3].get_ydata()
  s4 = b['whiskers'][4].get_ydata()
  e4 = b['whiskers'][5].get_ydata()
  s4c = b['whiskers'][6].get_ydata()
  e4c = b['whiskers'][7].get_ydata()
  
  print(f"{key}-VALUES:\n HM3D: BOTTOM {s3[1]:.4f} / MEDIAN {m3[0]:.4f} / TOPPER {e3[1]:.4f}\nHM3D-C: BOTTOM {s3c[1]:.4f} / MEDIAN {m3c[0]:.4f} / TOPPER {e3c[1]:.4f}\nHM4D: BOTTOM {s4[1]:.4f} / MEDIAN {m4[0]:.4f} / TOPPER {e4[1]:.4f}\nHM4D-C: BOTTOM {s4c[1]:.4f} / MEDIAN {m4c[0]:.4f} / TOPPER {e4c[1]:.4f}\n")
# fig.savefig('filename.png')
