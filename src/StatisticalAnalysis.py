#Import the required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sc

from scipy.stats import shapiro, mannwhitneyu, ttest_ind

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from statsmodels.graphics.gofplots import qqplot
from statsmodels.sandbox.stats.multicomp import multipletests

#Setup an excel writer to store all of the raw statistical data
writer = pd.ExcelWriter("excelfile.xlsx")

#Import the dataframe to analyze with the required group column
df = pd.read_csv("dataframe.csv")
groups_col = "Stage"
states = [x for x in df.Stage.unique()]
states = sorted(states)
banned = ['_pf_p_ix', '_pf_b_ix_l', '_pf_b_ix_r']
features = [feature for feature in df.columns if df[feature].dtype != object and '_pf_' in feature and feature not in banned]

#Get the sample size of every group
df.groupby(groups_col).count().to_excel(writer, sheet_name='occurance_count_per_stage')

#Setup a PDF file to store all data distribution QQ plots
pdf = PdfPages('pdffile.pdf')
d = pdf.infodict()
d['Title'] = 'All Untransformed Data Distributions from ... peakfeatures'
d['Author'] = 'My Name'
d['Subject'] = 'All used features and their corresponding qq plots'
d['Keywords'] = 'glutamate dynamics statistics qqplots matplotlib'

#Create all individual plots and end by saving the PDF
for feature in features:
    fig, axes = plt.subplots(4,2, figsize = (10,10))
    fig.subplots_adjust(left=0.25, wspace=0.3, hspace=0.3)
    axes = axes.ravel()
    axes[0].hist(df[feature], bins = 50);
    axes[0].set_title(f"All {feature} Data")
    axes[0].set_ylabel('Data point Recurrence')
    axes[0].set_xlabel('Data point value')
    fig.tight_layout()
    
    axes[1].set_title(f"Normalized {feature} histogram per stage")
    for state in states:
        axes[1].hist(df[feature][df[groups_col] == state], bins = 100, label = state, density = True, alpha = 0.5)
    axes[1].legend()
    axes[1].set_ylabel('Normalized Data point Recurrence')
    axes[1].set_xlabel('Data point value')
    fig.tight_layout()

    for state, ax in zip(states, axes[2:]):
        ax.set_title(f"{feature} - {state}")
        q = qqplot(df[feature][df[groups_col] == state], ax = ax, line = "s")
        fig.tight_layout()
        
    pdf.savefig(fig)
pdf.close()

#Store all testing plots collectively
pdf = PdfPages('pdffile.pdf')

d = pdf.infodict()
d['Title'] = 'All Untransformed Statistical tests from ... peakfeatures'
d['Author'] = 'My Name'
d['Subject'] = 'All plotted tests'
d['Keywords'] = 'glutamate dynamics statistics shapiro wilk mannwhitneyu holm bonferroni matplotlib'


#Apply the shapiro-wilk test
print("Shapiro Wilk tests")

pval_df = pd.DataFrame(data = states, columns = [groups_col])
for feature in features:
    result_per_state = []
    for state in states:
        stat, pval = shapiro(df[feature][df[groups_col] == state])
        result_per_state.append(pval)
    pval_df[feature] = result_per_state
pval_df.set_index(groups_col, inplace = True)
pval_dfc = pval_df.copy()
adjusted_pvals = pval_dfc.copy()

for section in pval_dfc.index:
    p_adjusted = multipletests(pval_dfc.loc[section].values, method='holm')
    p_adjusted_boolean = [1 if x==False else 0 for x in p_adjusted[0]]
    pval_dfc.loc[section] = p_adjusted_boolean
    adjusted_pvals.loc[section] = p_adjusted[1]

fig, ax = plt.subplots(figsize = (10,6))
im = ax.imshow(pval_dfc,vmin= 0, vmax= 1, cmap = "PiYG_r")
ax.set_ylim(-0.5, 4.5)
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(pval_dfc.index, rotation = "horizontal")
ax.set_xticks(range(0, len(features)))
ax.set_xticklabels(features, rotation = "vertical");

ax.vlines(np.arange(0.5, len(features)-0.5, 1), -0.5, 8, color = "w")
ax.hlines(np.arange(0.5, len(states)+0.5, 1), -0.5, len(features)-0.5, color = "w");

divider_ax = make_axes_locatable(ax)
cax = divider_ax.append_axes("right", size = "1%", pad = 0.1)
cb = plt.colorbar(im, cax = cax, orientation = "vertical")
cax.set_ylabel("pval colors (Boolean)")
ax.set_title("Holm Corrected pValues of Shapiro-Wilk Data")
fig.tight_layout(pad=1.5);

fig.savefig("save/your/plot.png");
fig.savefig("save/your/plot.svg");
pdf.savefig(fig)
adjusted_pvals.to_excel(writer, sheet_name='holm_shapirowilk_pVal')

#Apply the posthoc mannwhitney U-test
posthoc_results = {}
for feature in features:
    posthoc_results[feature] = sc.posthoc_mannwhitney(df, val_col=feature, group_col="Stage", p_adjust="holm")

df_extra = pd.concat({k: pd.DataFrame(v).T for k, v in posthoc_results.items()}, axis=0)
df_extra.to_excel(writer, sheet_name='holm_posthoc_mannwhitney_pVal')

letters = ['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I']
fig, axes = plt.subplots(3,3,figsize = (10,8), constrained_layout = True,)
for feature, ax, letter in zip(features, axes.ravel(), letters):
    pval_arr = posthoc_results[feature] > 0.05
    im = ax.imshow(pval_arr.astype(int), cmap = "PiYG_r")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_xlabel('Embryonic Stage')
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states)
    ax.set_ylabel('Embryonic Stage')
    ax.vlines(np.arange(0.5, len(states)+0.5, 1), -0.5, len(states)-0.5, color = "w", lw = 1.2)
    ax.hlines(np.arange(0.5, len(states)+0.5, 1), -0.5, len(states)-0.5, color = "w", lw = 1.2);
    ax.set_title(f"{letter} - {feature}")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad = "1%")
    plt.colorbar(im, cax=cax)
    cax.set_ylabel("p_value (Boolean)")

fig.savefig("save/your/plot.png")
fig.savefig("save/your/plot.svg")
pdf.savefig(fig)

#Apply the regular mannwhitney U-test
pval_df = pd.DataFrame(data = ['hm3D_CNO', 'hm4D_CNO'], columns = [groups_col])
for feature in features:
    result_per_state = []
    for state in [['hm3D_CNO', 'hm3D'], ['hm4D_CNO', 'hm4D']]:
        groups = state
        df_state = df[[feature, groups_col]][df[groups_col].isin(groups)]
        result = mannwhitneyu(*[df_state[feature][df_state[groups_col] == group].values for group in groups])
        result_per_state.append(result[1])
    pval_df[feature] = result_per_state
pval_df.set_index(groups_col, inplace = True)

df_wild = pd.DataFrame(data = ['Wild'], columns = [groups_col])
pval_df_w = pval_df.copy()

for feature in features:
    result_per_state = []
    groups = ['hm3D', 'hm4D']
    df_state = df[[feature, groups_col]][df[groups_col].isin(groups)]
    result = mannwhitneyu(*[df_state[feature][df_state[groups_col] == group].values for group in groups])
    result_per_state.append(result[1])
    df_wild[feature] = result_per_state
df_wild.set_index(groups_col, inplace = True)
pval_df_w = pval_df_w.append(df_wild)

pval_dfc_w = pval_df_w.copy()
adjusted_pvals = pval_dfc_w.copy()

for section in pval_dfc_w.index:
    p_adjusted = multipletests(pval_dfc_w.loc[section].values, method='holm')
    p_adjusted_boolean = [1 if x==False else 0 for x in p_adjusted[0]]
    pval_dfc_w.loc[section] = p_adjusted_boolean
    adjusted_pvals.loc[section] = p_adjusted[1]

fig, ax = plt.subplots(figsize = (10,6))
im = ax.imshow(pval_dfc_w, vmin= 0, vmax= 1, cmap = "PiYG_r")
ax.set_ylim(-0.5, 2.5)
ax.set_yticks([0,1,2])
ax.set_yticklabels(pval_dfc_w.index, rotation = "horizontal")
ax.set_xticks(range(0, len(features)))
ax.set_xticklabels(features, rotation = "vertical");

ax.vlines(np.arange(0.5, len(features)-0.5, 1), -0.5, 8, color = "w")
ax.hlines(np.arange(0.5, len(states)+0.5, 1), -0.5, len(features)-0.5, color = "w");

divider_ax = make_axes_locatable(ax)
cax = divider_ax.append_axes("right", size = "1%", pad = 0.1)
cb = plt.colorbar(im, cax = cax, orientation = "vertical")
cax.set_ylabel("pval colors (Boolean)")
ax.set_title("Holm Corrected MannwhitneyU pValues of ...")
fig.tight_layout(pad=1.5);

fig.savefig("save/your/plot.png");
fig.savefig("save/your/plot.svg");
pdf.savefig(fig)
adjusted_pvals.to_excel(writer, sheet_name='holm_mannwhitneyU_pVal')

#Apply the students ttest
features = [
    'from_center_max', 'from_center_mean', 'from_center_var',
    'turn005_mean','turn005_mean_at_speed', 'turn005_var','turn005_var_at_speed',
    'turn030_mean', 'turn030_mean_at_speed', 'turn030_var','turn030_var_at_speed',
    'totaldist', 'speed005_max', 'speed005_mean', 'speed005_var',
    'speed030_max', 'speed030_mean', 'speed030_var',
    'acceleration005_max', 'acceleration005_mean', 'acceleration005_var'
]

##The transformation
for feature in features:
    df[feature] = df[feature].transform(lambda x: np.sqrt(x))

mann_df = pd.DataFrame(data = ['cinuthm3d_CNO','cinuthm4d_CNO','cralbphm3d_CNO','cralbphm4d_CNO'], columns = [groups_col])
for feature in features:
    result_per_gene = []
    for gene in [['cinuthm3d_CNO','cinuthm3d'],
                 ['cinuthm4d_CNO','cinuthm4d'],
                 ['cralbphm3d_CNO','cralbphm3d'],
                 ['cralbphm4d_CNO','cralbphm4d']]:
        groups = gene
        df_gene = dfc[[feature, groups_col]][dfc[groups_col].isin(groups)]
        try:
            result = ttest_ind(*[df_gene[feature][df_gene[groups_col] == group].values for group in groups])
        except Exception as e:
            print(f"Error: {e}, in {gene} - {feature}")
            result = [0, 1]
        result_per_gene.append(result[1])
    mann_df[feature] = result_per_gene
mann_df.set_index(groups_col, inplace = True)

mann_dfc = mann_df.copy()
adjusted_pvals = mann_dfc.copy()

for section in mann_dfc.index:
    p_adjusted = multipletests(mann_dfc.loc[section].values, method='holm')
    p_adjusted_boolean = [1 if x==False else 0 for x in p_adjusted[0]]
    mann_dfc.loc[section] = p_adjusted_boolean
    adjusted_pvals.loc[section] = p_adjusted[1]
    
fig, ax = plt.subplots(figsize = (14,12))
im = ax.imshow(mann_dfc,vmin= 0, vmax= 1, cmap = "PiYG_r")
ax.set_ylim(-0.5, 3.5)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels(mann_dfc.index, rotation = "horizontal")
ax.set_xticks(range(0, len(features)))
ax.set_xticklabels(features, rotation = "vertical");

ax.vlines(np.arange(0.5, len(features)-0.5, 1), -0.5, 8, color = "w")
ax.hlines(np.arange(0.5, len(states)+0.5, 1), -0.5, len(features)-0.5, color = "w");

divider_ax = make_axes_locatable(ax)
cax = divider_ax.append_axes("right", size = "1%", pad = 0.1)
cb = plt.colorbar(im, cax = cax, orientation = "vertical")
cax.set_ylabel("pval colors (Boolean)")
ax.set_title("Holm Corrected Ttest pValues of ... Data")
fig.tight_layout();

fig.savefig("save/your/plot.png");
fig.savefig("save/your/plot.svg");
pdf.savefig(fig)
adjusted_pvals.to_excel(writer, sheet_name='holm_ttest_pVal')

#End by closing all openend storage files
writer.save()
pdf.close()
