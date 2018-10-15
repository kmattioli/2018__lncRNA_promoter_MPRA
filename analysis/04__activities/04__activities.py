
# coding: utf-8

# # 04__activities
# # analyzing activity levels per element (neg ctrls; between biotypes)
# 
# in this notebook, i perform analyses examining the activities of reference tiles in both pool1 and pool2. i compare reference sequences to negative controls, examine reference activities between biotypes, examine how MPRA activity compares to CAGE expression, and determine how many sequences are expressed across cell types.
# 
# ------
# 
# figures in this notebook:
# - **Fig 1C, Fig S4A, Fig S8**: boxplots comparing reference sequences to negative controls
# - **Fig 1D, Fig S4B**: boxplots comparing activities between biotypes
# - **Fig 1E**: KDE plot comparing CAGE cell type specificity and MPRA cell type specificity
# - **Fig 1F**: barplot showing % of reference sequences active in 1 or all 3 cell types

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time

from itertools import chain
from decimal import Decimal
from scipy import stats
from statsmodels.sandbox.stats import multicomp

# import utils
sys.path.append("../../utils")
from plotting_utils import *
from misc_utils import *
from norm_utils import *

get_ipython().magic('matplotlib inline')


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[3]:


activ_dir = "../../data/02__activs"
pval_dir = "../../data/03__pvals"
index_dir = "../../data/00__index"


# In[4]:


pool1_hela_barc_activ_f = "POOL1__pMPRA1__HeLa__activities_per_barcode.txt"
pool1_hepg2_barc_activ_f = "POOL1__pMPRA1__HepG2__activities_per_barcode.txt"
pool1_k562_barc_activ_f = "POOL1__pMPRA1__K562__activities_per_barcode.txt"

pool1_hela_elem_activ_f = "POOL1__pMPRA1__HeLa__activities_per_element.txt"
pool1_hepg2_elem_activ_f = "POOL1__pMPRA1__HepG2__activities_per_element.txt"
pool1_k562_elem_activ_f = "POOL1__pMPRA1__K562__activities_per_element.txt"

pool1_hela_pvals_f = "POOL1__pMPRA1__HeLa__pvals.txt"
pool1_hepg2_pvals_f = "POOL1__pMPRA1__HepG2__pvals.txt"
pool1_k562_pvals_f = "POOL1__pMPRA1__K562__pvals.txt"


# In[5]:


pool1_nocmv_hela_barc_activ_f = "POOL1__pNoCMVMPRA1__HeLa__activities_per_barcode.txt"
pool1_nocmv_hepg2_barc_activ_f = "POOL1__pNoCMVMPRA1__HepG2__activities_per_barcode.txt"
pool1_nocmv_k562_barc_activ_f = "POOL1__pNoCMVMPRA1__K562__activities_per_barcode.txt"

pool1_nocmv_hela_elem_activ_f = "POOL1__pNoCMVMPRA1__HeLa__activities_per_element.txt"
pool1_nocmv_hepg2_elem_activ_f = "POOL1__pNoCMVMPRA1__HepG2__activities_per_element.txt"
pool1_nocmv_k562_elem_activ_f = "POOL1__pNoCMVMPRA1__K562__activities_per_element.txt"

pool1_nocmv_hela_pvals_f = "POOL1__pNoCMVMPRA1__HeLa__pvals.txt"
pool1_nocmv_hepg2_pvals_f = "POOL1__pNoCMVMPRA1__HepG2__pvals.txt"
pool1_nocmv_k562_pvals_f = "POOL1__pNoCMVMPRA1__K562__pvals.txt"


# In[6]:


pool2_hepg2_barc_activ_f = "POOL2__pMPRA1__HepG2__activities_per_barcode.txt"
pool2_k562_barc_activ_f = "POOL2__pMPRA1__K562__activities_per_barcode.txt"

pool2_hepg2_elem_activ_f = "POOL2__pMPRA1__HepG2__activities_per_element.txt"
pool2_k562_elem_activ_f = "POOL2__pMPRA1__K562__activities_per_element.txt"

pool2_hepg2_pvals_f = "POOL2__pMPRA1__HepG2__pvals.txt"
pool2_k562_pvals_f = "POOL2__pMPRA1__K562__pvals.txt"


# In[7]:


pool1_index_f = "%s/tss_oligo_pool.index.txt" % index_dir
pool2_index_f = "%s/dels_oligo_pool.index.txt" % index_dir


# In[8]:


annot_f = "../../misc/00__tss_properties/mpra_id_to_biotype_map.txt"
id_map_f = "../../misc/00__tss_properties/mpra_tss_detailed_info.txt"
enh_id_map_f = "../../misc/00__tss_properties/enhancer_id_map.txt"
sel_map_f = "../../misc/00__tss_properties/mpra_tss_selection_info.txt"
cage_exp_f = "../../misc/other_files/All_TSS_and_enh.CAGE_grouped_exp.tissue_sp.txt"


# ## 1. import data

# In[9]:


pool1_index = pd.read_table(pool1_index_f, sep="\t")
pool2_index = pd.read_table(pool2_index_f, sep="\t")


# In[10]:


pool1_index_elem = pool1_index[["element", "oligo_type", "unique_id", "dupe_info", "SNP"]]
pool2_index_elem = pool2_index[["element", "oligo_type", "unique_id", "dupe_info", "SNP"]]

pool1_index_elem = pool1_index_elem.drop_duplicates()
pool2_index_elem = pool2_index_elem.drop_duplicates()


# In[11]:


annot = pd.read_table(annot_f, sep="\t")
annot.head()


# In[12]:


id_map = pd.read_table(id_map_f, sep="\t")
id_map.head()


# In[13]:


sel_map = pd.read_table(sel_map_f, sep="\t")
sel_map.head()


# In[14]:


enh_id_map = pd.read_table(enh_id_map_f, sep="\t")
enh_id_map.head()


# In[15]:


cage_exp = pd.read_table(cage_exp_f, sep="\t")
cage_exp.head()


# ### pool 1

# In[16]:


pool1_hela_elem_norm = pd.read_table("%s/%s" % (activ_dir, pool1_hela_elem_activ_f), sep="\t")
pool1_hepg2_elem_norm = pd.read_table("%s/%s" % (activ_dir, pool1_hepg2_elem_activ_f), sep="\t")
pool1_k562_elem_norm = pd.read_table("%s/%s" % (activ_dir, pool1_k562_elem_activ_f), sep="\t")
pool1_hepg2_elem_norm.head()


# In[17]:


pool1_hela_reps = [x for x in pool1_hela_elem_norm.columns if "rna_" in x]
pool1_hepg2_reps = [x for x in pool1_hepg2_elem_norm.columns if "rna_" in x]
pool1_k562_reps = [x for x in pool1_k562_elem_norm.columns if "rna_" in x]
pool1_hepg2_reps


# In[18]:


pool1_hela_barc_norm = pd.read_table("%s/%s" % (activ_dir, pool1_hela_barc_activ_f), sep="\t")
pool1_hepg2_barc_norm = pd.read_table("%s/%s" % (activ_dir, pool1_hepg2_barc_activ_f), sep="\t")
pool1_k562_barc_norm = pd.read_table("%s/%s" % (activ_dir, pool1_k562_barc_activ_f), sep="\t")
pool1_hepg2_barc_norm.head()


# In[19]:


pool1_hela_pvals = pd.read_table("%s/%s" % (pval_dir, pool1_hela_pvals_f), sep="\t")
pool1_hepg2_pvals = pd.read_table("%s/%s" % (pval_dir, pool1_hepg2_pvals_f), sep="\t")
pool1_k562_pvals = pd.read_table("%s/%s" % (pval_dir, pool1_k562_pvals_f), sep="\t")
pool1_hepg2_pvals.head()


# ### pool 2

# In[20]:


pool2_hepg2_elem_norm = pd.read_table("%s/%s" % (activ_dir, pool2_hepg2_elem_activ_f), sep="\t")
pool2_k562_elem_norm = pd.read_table("%s/%s" % (activ_dir, pool2_k562_elem_activ_f), sep="\t")
pool2_hepg2_elem_norm.head()


# In[21]:


pool2_hepg2_reps = [x for x in pool2_hepg2_elem_norm.columns if "rna_" in x]
pool2_k562_reps = [x for x in pool2_k562_elem_norm.columns if "rna_" in x]
pool2_hepg2_reps


# In[22]:


pool2_hepg2_barc_norm = pd.read_table("%s/%s" % (activ_dir, pool2_hepg2_barc_activ_f), sep="\t")
pool2_k562_barc_norm = pd.read_table("%s/%s" % (activ_dir, pool2_k562_barc_activ_f), sep="\t")
pool2_hepg2_barc_norm.head()


# In[23]:


pool2_hepg2_pvals = pd.read_table("%s/%s" % (pval_dir, pool2_hepg2_pvals_f), sep="\t")
pool2_k562_pvals = pd.read_table("%s/%s" % (pval_dir, pool2_k562_pvals_f), sep="\t")
pool2_hepg2_pvals.head()


# ## 2. merge with index

# ### pool 1

# In[24]:


pool1_hela_elem_norm = pool1_hela_elem_norm.merge(pool1_index_elem, on=["unique_id", "element"], how="left")
pool1_hepg2_elem_norm = pool1_hepg2_elem_norm.merge(pool1_index_elem, on=["unique_id", "element"], how="left")
pool1_k562_elem_norm = pool1_k562_elem_norm.merge(pool1_index_elem, on=["unique_id", "element"], how="left")


# In[25]:


pool1_hela_barc_norm = pool1_hela_barc_norm.merge(pool1_index, left_on="barcode", right_on="barcode", how="left")
pool1_hepg2_barc_norm = pool1_hepg2_barc_norm.merge(pool1_index, left_on="barcode", right_on="barcode", how="left")
pool1_k562_barc_norm = pool1_k562_barc_norm.merge(pool1_index, left_on="barcode", right_on="barcode", how="left")


# In[26]:


pool1_hela_elem_norm["better_type"] = pool1_hela_elem_norm.apply(better_type, axis=1)
pool1_hepg2_elem_norm["better_type"] = pool1_hepg2_elem_norm.apply(better_type, axis=1)
pool1_k562_elem_norm["better_type"] = pool1_k562_elem_norm.apply(better_type, axis=1)


# In[27]:


pool1_hela_elem_norm = pool1_hela_elem_norm.merge(pool1_hela_pvals.drop("oligo_type", axis=1), on=["unique_id", "element"], how="left")
pool1_hepg2_elem_norm = pool1_hepg2_elem_norm.merge(pool1_hepg2_pvals.drop("oligo_type", axis=1), on=["unique_id", "element"], how="left")
pool1_k562_elem_norm = pool1_k562_elem_norm.merge(pool1_k562_pvals.drop("oligo_type", axis=1), on=["unique_id", "element"], how="left")
pool1_hepg2_elem_norm.head()


# ### pool 2

# In[28]:


pool2_hepg2_elem_norm = pool2_hepg2_elem_norm.merge(pool2_index_elem, on=["unique_id", "element"], how="left")
pool2_k562_elem_norm = pool2_k562_elem_norm.merge(pool2_index_elem, on=["unique_id", "element"], how="left")


# In[29]:


pool2_hepg2_barc_norm = pool2_hepg2_barc_norm.merge(pool2_index, left_on="barcode", right_on="barcode", how="left")
pool2_k562_barc_norm = pool2_k562_barc_norm.merge(pool2_index, left_on="barcode", right_on="barcode", how="left")


# In[30]:


pool2_hepg2_elem_norm["better_type"] = pool2_hepg2_elem_norm.apply(better_type, axis=1)
pool2_k562_elem_norm["better_type"] = pool2_k562_elem_norm.apply(better_type, axis=1)


# In[31]:


pool2_hepg2_elem_norm = pool2_hepg2_elem_norm.merge(pool2_hepg2_pvals.drop("oligo_type", axis=1), on=["unique_id", "element"], how="left")
pool2_k562_elem_norm = pool2_k562_elem_norm.merge(pool2_k562_pvals.drop("oligo_type", axis=1), on=["unique_id", "element"], how="left")
pool2_hepg2_elem_norm.head()


# ## 3. count significantly active/inactive tiles

# ### pool 1

# In[32]:


pool1_hela_elem_norm["overall_mean"] = pool1_hela_elem_norm[pool1_hela_reps].mean(axis=1)
pool1_hepg2_elem_norm["overall_mean"] = pool1_hepg2_elem_norm[pool1_hepg2_reps].mean(axis=1)
pool1_k562_elem_norm["overall_mean"] = pool1_k562_elem_norm[pool1_k562_reps].mean(axis=1)

pool1_hela_elem_norm["overall_median"] = pool1_hela_elem_norm[pool1_hela_reps].median(axis=1)
pool1_hepg2_elem_norm["overall_median"] = pool1_hepg2_elem_norm[pool1_hepg2_reps].median(axis=1)
pool1_k562_elem_norm["overall_median"] = pool1_k562_elem_norm[pool1_k562_reps].median(axis=1)


# In[33]:


for cell, df in zip(["HeLa", "HepG2", "K562"], [pool1_hela_elem_norm, pool1_hepg2_elem_norm, pool1_k562_elem_norm]):
    print("%s: combined class" % cell)
    print(df.combined_class.value_counts())
    print("")
    if cell == "HepG2":
        print("%s: downsampled class" % cell)
        print(df.downsamp_combined_class.value_counts())
        print("")


# ### pool 2

# In[34]:


pool2_hepg2_elem_norm["overall_mean"] = pool2_hepg2_elem_norm[pool2_hepg2_reps].mean(axis=1)
pool2_k562_elem_norm["overall_mean"] = pool2_k562_elem_norm[pool2_k562_reps].mean(axis=1)

pool2_hepg2_elem_norm["overall_median"] = pool2_hepg2_elem_norm[pool2_hepg2_reps].median(axis=1)
pool2_k562_elem_norm["overall_median"] = pool2_k562_elem_norm[pool2_k562_reps].median(axis=1)


# In[35]:


for cell, df in zip(["HepG2", "K562"], [pool2_hepg2_elem_norm, pool2_k562_elem_norm]):
    print("%s: combined class" % cell)
    print(df.combined_class.value_counts())
    print("")
    if cell == "HepG2":
        print("%s: downsampled class" % cell)
        print(df.downsamp_combined_class.value_counts())
        print("")


# ## 4. boxplots: neg ctrls vs reference

# In[36]:


pool1_hepg2_df = pool1_hepg2_elem_norm.merge(annot, left_on="unique_id", right_on="seqID", how="left")
pool1_hela_df = pool1_hela_elem_norm.merge(annot, left_on="unique_id", right_on="seqID", how="left")
pool1_k562_df = pool1_k562_elem_norm.merge(annot, left_on="unique_id", right_on="seqID", how="left")
pool1_hepg2_df.head()


# In[37]:


pool1_hepg2_df["oligo_reg"] = pool1_hepg2_df.unique_id.str.split("__", expand=True)[2]
pool1_hela_df["oligo_reg"] = pool1_hela_df.unique_id.str.split("__", expand=True)[2]
pool1_k562_df["oligo_reg"] = pool1_k562_df.unique_id.str.split("__", expand=True)[2]
pool1_hepg2_df.head()


# In[38]:


def add_neg_ctrl_promtype(row):
    if row["better_type"] == "RANDOM":
        return "random"
    elif row["better_type"] == "SCRAMBLED":
        return "scrambled"
    elif row["better_type"] == "CONTROL":
        return "control"
    else:
        return row["PromType2"]


# In[39]:


pool1_hepg2_df["PromType2"] = pool1_hepg2_df.apply(add_neg_ctrl_promtype, axis=1)
pool1_hela_df["PromType2"] = pool1_hela_df.apply(add_neg_ctrl_promtype, axis=1)
pool1_k562_df["PromType2"] = pool1_k562_df.apply(add_neg_ctrl_promtype, axis=1)
pool1_hepg2_df.sample(10)


# ### pool 1

# In[40]:


def annotate_pval(ax, x1, x2, y, h, text_y, val, fontsize, mark_points, color1, color2):
    from decimal import Decimal
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1, c="black", linewidth=0.5)
    if mark_points:
        ax.plot(x1, y, 's', markersize=5, markerfacecolor='white', markeredgewidth=1, markeredgecolor=color1)
        ax.plot(x2, y, 's', markersize=5, markerfacecolor='white', markeredgewidth=1, markeredgecolor=color2)
    if val < 0.0005:
        text = "{:.1e}".format(Decimal(val))
        #text = "**"
    elif val < 0.05:
        text = "%.3f" % val
        #text = "*"
    else:
        text = "%.3f" % val
        #text = "n.s."
        
    ax.annotate(text, xy=((x1+x2)*.5, y), xycoords="data", xytext=(0, text_y), textcoords="offset pixels",
                horizontalalignment="center", verticalalignment="bottom", color="black", size=fontsize)
    
    # ax.text((x1+x2)*.5, text_y, text, ha='center', va='bottom', color="black", size=fontsize)


# In[41]:


def neg_control_plot(df, order, palette, fontsize, cell_type, ax, figsize, ylabel, sharey, title, save, plotname):
    df_sub = df[df["better_type"].isin(["WILDTYPE", "RANDOM", "SCRAMBLED"])].drop_duplicates()
    
    if ax == None:
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df_sub, x="better_type", y="overall_mean", order=order, palette=palette, linewidth=1,
                         saturation=1, flierprops = dict(marker='o', markersize=5))
    else:
        sns.boxplot(data=df_sub, x="better_type", y="overall_mean", order=order, palette=palette, linewidth=1,
                    saturation=1, flierprops = dict(marker='o', markersize=5), ax=ax)

    mimic_r_boxplot(ax)

    # calc p-vals b/w dists
    rand_dist = np.asarray(df[df["better_type"] == "RANDOM"]["overall_mean"])
    scram_dist = np.asarray(df[df["better_type"] == "SCRAMBLED"]["overall_mean"])
    wt_dist = np.asarray(df[df["better_type"] == "WILDTYPE"]["overall_mean"])

    rand_dist = rand_dist[~np.isnan(rand_dist)]
    scram_dist = scram_dist[~np.isnan(scram_dist)]
    wt_dist = wt_dist[~np.isnan(wt_dist)]

    rand_u, rand_pval = stats.mannwhitneyu(rand_dist, wt_dist, alternative="two-sided", use_continuity=False)
    scram_u, scram_pval = stats.mannwhitneyu(scram_dist, wt_dist, alternative="two-sided", use_continuity=False)
    
    if sharey:
        ax.set_ylim((-12, 8))
        # ax.yaxis.set_ticks(np.arange(-15, 11, 5))
        y_2 = 6 # set lowest one

    else:
        ax.set_ylim((np.min(rand_dist)-4.5, np.max(wt_dist)+4.5))
        y_2 = np.max(wt_dist)+2.5 # set lowest one

    # find y_1 by going up from y_2 in axes fraction coords
    x_ax, y_ax = axis_data_coords_sys_transform(ax, 0, y_2, inverse=True)
    y_1_ax = y_ax + 0.125
    x_data, y_1 = axis_data_coords_sys_transform(ax, x_ax, y_1_ax, inverse=False)
    print("y_1: %s, y_2: %s" % (y_1, y_2))
    print("rand_pval: %s, scram_pval: %s" % (rand_pval, scram_pval))
    
    # reset axlim
    if not sharey:
        x_ax, ylim_ax = axis_data_coords_sys_transform(ax, 0, y_1_ax + 0.15, inverse=False)
        ax.set_ylim((np.min(rand_dist)-4.5, ylim_ax))

    # statistical annotation and group numbers
    x_ax_0, y_ax = axis_data_coords_sys_transform(ax, 0, 0, inverse=True)
    x_ax_1, y_ax = axis_data_coords_sys_transform(ax, 1, 0, inverse=True)
    x_ax_2, y_ax = axis_data_coords_sys_transform(ax, 2, 0, inverse=True)

    if len(order) == 3:
        annotate_pval(ax, 0, 2, y_1, 0, 0, rand_pval, fontsize, False, None, None)
        annotate_pval(ax, 1, 2, y_2, 0, 0, scram_pval, fontsize, False, None, None)
        
        ax.annotate(str(len(rand_dist)), xy=(x_ax_0, 0.2), xycoords="axes fraction", xytext=(0, -15), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["RANDOM"], size=fontsize)
        
        ax.annotate(str(len(scram_dist)), xy=(x_ax_1, 0.2), xycoords="axes fraction", xytext=(0, -15), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["SCRAMBLED"], size=fontsize)
        
        ax.annotate(str(len(wt_dist)), xy=(x_ax_2, 0.2), xycoords="axes fraction", xytext=(0, -15), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["WILDTYPE"], size=fontsize)
        ax.set_xticklabels(["random", "scrambled", "core promoters"], rotation=30)
    elif len(order) == 2:
        annotate_pval(ax, 0, 1, y_2, 0, 0, rand_pval, fontsize, False, None, None)
        ax.annotate(str(len(rand_dist)), xy=(x_ax_0, 0.2), xycoords="axes fraction", xytext=(0, -15), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["RANDOM"], size=fontsize)
        ax.annotate(str(len(wt_dist)), xy=(x_ax_1, 0.2), xycoords="axes fraction", xytext=(0, -15), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["WILDTYPE"], size=fontsize)
        ax.set_xticklabels(["random seqs", "core promoters"], rotation=30)
        
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if title:
        ax.set_title("%s" % (cell_type))
    if save:
        plt.savefig("%s/%s.pdf" % (figs_dir, plotname), dpi="figure", bbox_inches="tight")


# In[42]:


def axis_data_coords_sys_transform(axis_obj_in,xin,yin,inverse=False):
    """ inverse = False : Axis => Data
                = True  : Data => Axis
    """
    xlim = axis_obj_in.get_xlim()
    ylim = axis_obj_in.get_ylim()

    xdelta = xlim[1] - xlim[0]
    ydelta = ylim[1] - ylim[0]
    if not inverse:
        xout =  xlim[0] + xin * xdelta
        yout =  ylim[0] + yin * ydelta
    else:
        xdelta2 = xin - xlim[0]
        ydelta2 = yin - ylim[0]
        xout = xdelta2 / xdelta
        yout = ydelta2 / ydelta
    return xout,yout


# In[43]:


order = ["RANDOM", "SCRAMBLED", "WILDTYPE"]
palette = {"RANDOM": "gray", "SCRAMBLED": "gray", "WILDTYPE": "black"}

f, axarr = plt.subplots(3, sharex=True, sharey=False, figsize=(1.78, 5))
neg_control_plot(pool1_hela_elem_norm, order, palette, fontsize, "HeLa", axarr[0], None, "HeLa MPRA activity", 
                 False, False, False, None)
neg_control_plot(pool1_hepg2_elem_norm, order, palette, fontsize, "HepG2", axarr[1], None, "HepG2 MPRA activity", 
                 False, False, False, None)
neg_control_plot(pool1_k562_elem_norm, order, palette, fontsize, "K562", axarr[2], None, "K562 MPRA activity", 
                 False, False, False, None)
plt.tight_layout()
f.savefig("Fig_1C_S4A.pdf", bbox_inches="tight", dpi="figure")


# In[44]:


talk_order = ["RANDOM", "WILDTYPE"]
talk_palette = {"RANDOM": "gray", "WILDTYPE": "black"}

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(4.7, 2))
neg_control_plot(pool1_hela_elem_norm, talk_order, talk_palette, fontsize, "HeLa", axarr[0], None, "HeLa MPRA activity", 
                 True, False, False, None)
neg_control_plot(pool1_hepg2_elem_norm, talk_order, talk_palette, fontsize, "HepG2", axarr[1], None, "HepG2 MPRA activity", 
                 True, False, False, None)
neg_control_plot(pool1_k562_elem_norm, talk_order, talk_palette, fontsize, "K562", axarr[2], None, "K562 MPRA activity", 
                 True, False, False, None)
plt.tight_layout()
#plt.ylim((-10, 11))
f.savefig("neg_ctrl_boxplots.for_talk.pdf", bbox_inches="tight", dpi="figure")


# ### pool 2

# In[45]:


f, axarr = plt.subplots(2, sharex=True, sharey=False, figsize=(1.78, 3.2))
neg_control_plot(pool2_hepg2_elem_norm, order, palette, fontsize, "HepG2", axarr[0], None, "HepG2 MPRA activity", 
                 False, False, False, None)
neg_control_plot(pool2_k562_elem_norm, order, palette, fontsize, "K562", axarr[1], None, "K562 MPRA activity", 
                 False, False, False, None)
plt.tight_layout()
f.savefig("Fig_S8.pdf", bbox_inches="tight", dpi="figure")


# ## 5. boxplots: across TSS classes

# In[46]:


def promtype_plot(df, order, palette, labels, fontsize, cell_type, ax, figsize, ylabel, sharey, title, save, plotname):
    
    df = df[df["better_type"].isin(["WILDTYPE", "SCRAMBLED", "RANDOM"])]
    
    if ax == None:
        plt.figure(figsize=figsize)
        ax = sns.boxplot(data=df, x="PromType2", y="overall_mean", order=order, palette=palette, linewidth=1,
                         saturation=1, flierprops=dict(marker='o', markersize=5))
    else:
        sns.boxplot(ax=ax, data=df, x="PromType2", y="overall_mean", order=order, palette=palette, linewidth=1,
                    saturation=1, flierprops=dict(marker='o', markersize=5))
    
    if "random" in order:
        ax.set_xticklabels(["random", "eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"], rotation=30)
    elif "scrambled" in order:
        ax.set_xticklabels(["scrambled", "eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"], rotation=30)
    else:
        ax.set_xticklabels(["eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"], rotation=30)
    mimic_r_boxplot(ax)
    
    # calc p-vals b/w dists
    if "random" in order:
        rand_dist = np.asarray(df[df["PromType2"] == "random"]["overall_mean"])
        rand_dist = rand_dist[~np.isnan(rand_dist)]
    if "scrambled" in order:
        scram_dist = np.asarray(df[df["PromType2"] == "scrambled"]["overall_mean"])
        scram_dist = scram_dist[~np.isnan(scram_dist)]
        
    enh_dist = np.asarray(df[df["PromType2"] == "Enhancer"]["overall_mean"])
    enh_dist = enh_dist[~np.isnan(enh_dist)]
    
    int_dist = np.asarray(df[df["PromType2"] == "intergenic"]["overall_mean"])
    int_dist = int_dist[~np.isnan(int_dist)]
    
    div_lnc_dist = np.asarray(df[df["PromType2"] == "div_lnc"]["overall_mean"])
    div_lnc_dist = div_lnc_dist[~np.isnan(div_lnc_dist)]
    
    pc_dist = np.asarray(df[df["PromType2"] == "protein_coding"]["overall_mean"])
    pc_dist = pc_dist[~np.isnan(pc_dist)]
    
    div_pc_dist = np.asarray(df[df["PromType2"] == "div_pc"]["overall_mean"])
    div_pc_dist = div_pc_dist[~np.isnan(div_pc_dist)]
    
    if "random" in order:
        # random pvals
        enh_n_u, enh_n_pval = stats.mannwhitneyu(rand_dist, enh_dist, alternative="two-sided", use_continuity=False)
        int_n_u, int_n_pval = stats.mannwhitneyu(rand_dist, int_dist, alternative="two-sided", use_continuity=False)
        div_lnc_n_u, div_lnc_n_pval = stats.mannwhitneyu(rand_dist, div_lnc_dist, alternative="two-sided", use_continuity=False)
        pc_n_u, pc_n_pval = stats.mannwhitneyu(rand_dist, pc_dist, alternative="two-sided", use_continuity=False)
        div_pc_n_u, div_pc_n_pval = stats.mannwhitneyu(rand_dist, div_pc_dist, alternative="two-sided", use_continuity=False)
    
    if "scrambled" in order:
        # scrambled pvals
        enh_n_u, enh_n_pval = stats.mannwhitneyu(scram_dist, enh_dist, alternative="two-sided", use_continuity=False)
        int_n_u, int_n_pval = stats.mannwhitneyu(scram_dist, int_dist, alternative="two-sided", use_continuity=False)
        div_lnc_n_u, div_lnc_n_pval = stats.mannwhitneyu(scram_dist, div_lnc_dist, alternative="two-sided", use_continuity=False)
        pc_n_u, pc_n_pval = stats.mannwhitneyu(scram_dist, pc_dist, alternative="two-sided", use_continuity=False)
        div_pc_n_u, div_pc_n_pval = stats.mannwhitneyu(scram_dist, div_pc_dist, alternative="two-sided", use_continuity=False)
    
    lnc_u, lnc_pval = stats.mannwhitneyu(int_dist, div_lnc_dist, alternative="two-sided", use_continuity=False)
    pc_u, pc_pval = stats.mannwhitneyu(pc_dist, div_pc_dist, alternative="two-sided", use_continuity=False)

    if "random" in order:
        all_dists = list(rand_dist) + list(enh_dist) + list(int_dist) + list(div_lnc_dist) + list(pc_dist) + list(div_pc_dist)
    if "scrambled" in order:
        all_dists = list(scram_dist) + list(enh_dist) + list(int_dist) + list(div_lnc_dist) + list(pc_dist) + list(div_pc_dist)
    else:
        all_dists = list(enh_dist) + list(int_dist) + list(div_lnc_dist) + list(pc_dist) + list(div_pc_dist)
        
    if sharey:
        ax.set_ylim((-20, 10))
        #ax.yaxis.set_ticks(np.arange(-15, 11, 5))
        y_2 = 6
        y_1 = 6
    else:
        ax.set_ylim((np.min(all_dists)-5, np.max(all_dists)+12))
        #ax.yaxis.set_ticks(np.arange(round(np.min(all_dists)-2), round(np.max(all_dists)+3.5), 5))
        y_2 = np.max(pc_dist)+2
        y_1 = np.max(pc_dist)+2
    
        
    # statistical annotation for divergents
    if "random" in order or "scrambled" in order:
        annotate_pval(ax, 2, 3, y_1, 0, 0, lnc_pval, fontsize, True,
                      palette["intergenic"], palette["div_lnc"])
        annotate_pval(ax, 4, 5, y_2, 0, 0, pc_pval, fontsize, True,
                      palette["protein_coding"], palette["div_pc"])
    else:
        annotate_pval(ax, 1, 2, y_1, 0, 0, lnc_pval, fontsize, True,
                      palette["intergenic"], palette["div_lnc"])
        annotate_pval(ax, 3, 4, y_2, 0, 0, pc_pval, fontsize, True,
                      palette["protein_coding"], palette["div_pc"])
    
    # statistical annotation for nulls
    if "random" in order or "scrambled" in order:
        y_1 = np.max(all_dists)+2
        annotate_pval(ax, 0, 1, y_1, 0, 0, enh_n_pval, fontsize, True, 
                      palette["random"], palette["Enhancer"])
        for i, color, p in zip([2,3,4,5],[palette["intergenic"], palette["div_lnc"], palette["protein_coding"], palette["div_pc"]], [int_n_pval, div_lnc_n_pval, pc_n_pval, div_pc_n_pval]):
            # find y_1 by going up from y_2 in axes fraction coords
            x_ax, y_ax = axis_data_coords_sys_transform(ax, 0, y_1, inverse=True)
            y_1_ax = y_ax + 0.08
            x_data, y_1 = axis_data_coords_sys_transform(ax, x_ax, y_1_ax, inverse=False)

            annotate_pval(ax, 0, i, y_1, 0, 0, p, fontsize, True,
                          palette["random"], color)
        
    # reset axlim
    if not sharey:
        x_ax, y_ax = axis_data_coords_sys_transform(ax, 0, y_1, inverse=True)
        x_d, ylim_d = axis_data_coords_sys_transform(ax, 0, y_ax + 0.08, inverse=False)
        ax.set_ylim((np.min(all_dists)-5, ylim_d))
    
    # annotate group #s
    x_ax_0, y_ax = axis_data_coords_sys_transform(ax, 0, 0, inverse=True)
    x_ax_1, y_ax = axis_data_coords_sys_transform(ax, 1, 0, inverse=True)
    x_ax_2, y_ax = axis_data_coords_sys_transform(ax, 2, 0, inverse=True)
    x_ax_3, y_ax = axis_data_coords_sys_transform(ax, 3, 0, inverse=True)
    x_ax_4, y_ax = axis_data_coords_sys_transform(ax, 4, 0, inverse=True)
    x_ax_5, y_ax = axis_data_coords_sys_transform(ax, 5, 0, inverse=True)
    
    
    if "random" in order:
        ax.annotate(str(len(rand_dist)), xy=(x_ax_0, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["random"], size=fontsize)
    if "scrambled" in order:
        ax.annotate(str(len(scram_dist)), xy=(x_ax_0, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                    textcoords="offset pixels", ha='center', va='bottom', 
                    color=palette["scrambled"], size=fontsize)
        
    if "random" not in order and "scrambled" not in order:
        diff = 1/len(order)
    else:
        diff = 0
    ax.annotate(str(len(enh_dist)), xy=(x_ax_1-diff, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=palette["Enhancer"], size=fontsize)
    ax.annotate(str(len(int_dist)), xy=(x_ax_2-diff, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=palette["intergenic"], size=fontsize)
    ax.annotate(str(len(div_lnc_dist)), xy=(x_ax_3-diff, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=palette["div_lnc"], size=fontsize)
    ax.annotate(str(len(pc_dist)), xy=(x_ax_4-diff, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=palette["protein_coding"], size=fontsize)
    ax.annotate(str(len(div_pc_dist)), xy=(x_ax_5-diff, 0.02), xycoords="axes fraction", xytext=(0, 0), 
                textcoords="offset pixels", ha='center', va='bottom', 
                color=palette["div_pc"], size=fontsize)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if title:
        ax.set_title("%s" % (cell_type))
    if save:
        plt.savefig("%s/%s.pdf" % (figs_dir, plotname), dpi="figure", bbox_inches="tight")


# In[47]:


palette = {"random": "gray", "scrambled": "gray", "Enhancer": sns.color_palette("deep")[1], 
           "intergenic": sns.color_palette("deep")[2], "protein_coding": sns.color_palette("deep")[5], 
           "div_lnc": sns.color_palette("deep")[3], "div_pc": sns.color_palette("deep")[0]}


# In[48]:


# random
order = ["random", "Enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]
labels = ["random", "eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"]

f, axarr = plt.subplots(3, sharex=True, sharey=False, figsize=(4, 8))
promtype_plot(pool1_hela_df, order, palette, labels, fontsize, "HeLa", axarr[0], None, 
              "HeLa MPRA activity", False, False, False, None)
promtype_plot(pool1_hepg2_df, order, palette, labels, fontsize, "HepG2", axarr[1], None, 
              "HepG2 MPRA activity", False, False, False, None)
promtype_plot(pool1_k562_df, order, palette, labels, fontsize, "K562", axarr[2], None, 
              "K562 MPRA activity", False, False, False, None)
plt.tight_layout()
f.savefig("Fig1_All_Biotypes_v_Random.pdf", bbox_inches="tight", dpi="figure")


# In[49]:


# scrambled
order = ["scrambled", "Enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]
labels = ["scrambled", "eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"]

f, axarr = plt.subplots(3, sharex=True, sharey=False, figsize=(4, 8))
promtype_plot(pool1_hela_df, order, palette, labels, fontsize, "HeLa", axarr[0], None, 
              "HeLa MPRA activity", False, False, False, None)
promtype_plot(pool1_hepg2_df, order, palette, labels, fontsize, "HepG2", axarr[1], None, 
              "HepG2 MPRA activity", False, False, False, None)
promtype_plot(pool1_k562_df, order, palette, labels, fontsize, "K562", axarr[2], None, 
              "K562 MPRA activity", False, False, False, None)
plt.tight_layout()
f.savefig("Fig1_All_Biotypes_v_Scrambled.pdf", bbox_inches="tight", dpi="figure")


# ## expression-match

# In[50]:


def fix_enh_cage_id(row):
    if row.PromType2 == "Enhancer":
        if not pd.isnull(row.enhancer_id_x):
            return row.enhancer_id_x
        elif not pd.isnull(row.enhancer_id_y):
            return row.enhancer_id_y
        else:
            return np.nan
    else:
        return row.cage_id


# In[51]:


sel_map["cage_id"] = sel_map["TSS_id"]
sel_map = sel_map.merge(enh_id_map[["TSS_id_Pos", "enhancer_id"]],
                        left_on="TSS_id", right_on="TSS_id_Pos", how="left")
sel_map = sel_map.merge(enh_id_map[["TSS_id_Neg", "enhancer_id"]],
                        left_on="TSS_id", right_on="TSS_id_Neg", how="left")
sel_map["cage_id"] = sel_map.apply(fix_enh_cage_id, axis=1)
sel_map.drop(["TSS_id_Pos", "enhancer_id_x", "TSS_id_Neg", "enhancer_id_y"], axis=1, inplace=True)
sel_map_expr = sel_map.merge(cage_exp, on="cage_id", how="left")
sel_map_expr.sample(5)


# In[52]:


sel_map_expr["log_av_exp"] = np.log10(sel_map_expr["av_exp"])
sel_map_expr.selected.unique()


# In[53]:


rand_sel_types = ["lncRNA", "eRNA.random", "mRNA.random", "mRNA.bidirec70-160"]


# In[54]:


rand_sel_ids = sel_map_expr[sel_map_expr["selected"].isin(rand_sel_types)]
remaining_ids = sel_map_expr[~sel_map_expr["selected"].isin(rand_sel_types)]
remaining_ids = remaining_ids.append(sel_map_expr[sel_map_expr["selected"] == "lncRNA"])
remaining_ids.selected.value_counts()


# In[55]:


def distplot_biotypes(df, figsize, palette, label_dict, ylim, xlabel, save, plotname):
    fig = plt.figure(figsize=figsize)
    df = df.drop_duplicates()
    
    for i, promtype in enumerate(["Enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]):
        sub = df[df["PromType2"] == promtype]
        color = palette[promtype]
        label = label_dict[promtype]
        if i == 0:
            ax = sns.kdeplot(sub["log_av_exp"], cumulative=True, color=color, 
                              label="%s (n=%s)" % (label, len(sub)))
        else:
            sns.kdeplot(sub["log_av_exp"], cumulative=True, color=color, 
                         label="%s (n=%s)" % (label, len(sub)), ax=ax)
    
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel("cumulative density")
    if save:
        fig.savefig("%s.pdf" % plotname, dpi="figure", bbox_inches="tight")


# In[56]:


label_dict = {"Enhancer": "eRNAs", "intergenic": "lincRNAs", "div_lnc": "div. lncRNAs", "protein_coding": "mRNAs",
              "div_pc": "div. mRNAs"}
distplot_biotypes(rand_sel_ids, (3, 2.5), palette, label_dict, (0, 1.01), "log10(average CAGE expression)", True, "Random_Sel_Expr_Dist")


# In[57]:


remaining_ids.PromType2.value_counts()


# In[58]:


distplot_biotypes(remaining_ids, (3, 2.5), palette, label_dict, (0, 1.01), "log10(average CAGE expression)", False, None)


# In[59]:


exp_match_ids = remaining_ids[(remaining_ids["log_av_exp"] > 0) & (remaining_ids["log_av_exp"] < 1.5)]
print("min exp: %s" % (np.min(exp_match_ids["av_exp"])))
print("max exp: %s" % (np.max(exp_match_ids["av_exp"])))
exp_match_ids.PromType2.value_counts()


# In[60]:


distplot_biotypes(exp_match_ids, (3, 2.5), palette, label_dict, (0, 1.01), "log10(average CAGE expression)", True, "ExpMatch_Sel_Expr_Dist")


# In[61]:


pool1_hela_rand = pool1_hela_df[pool1_hela_df["oligo_reg"].isin(rand_sel_ids["oligo_reg"])]
pool1_hepg2_rand = pool1_hepg2_df[pool1_hepg2_df["oligo_reg"].isin(rand_sel_ids["oligo_reg"])]
pool1_k562_rand = pool1_k562_df[pool1_k562_df["oligo_reg"].isin(rand_sel_ids["oligo_reg"])]


# In[62]:


order = ["Enhancer", "intergenic", "div_lnc", "protein_coding", "div_pc"]
labels = ["eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"]

f, axarr = plt.subplots(3, sharex=True, sharey=False, figsize=(4, 8))
promtype_plot(pool1_hela_rand, order, palette, labels, fontsize, "HeLa", axarr[0], None, 
              "HeLa MPRA activity", False, False, False, None)
promtype_plot(pool1_hepg2_rand, order, palette, labels, fontsize, "HepG2", axarr[1], None, 
              "HepG2 MPRA activity", False, False, False, None)
promtype_plot(pool1_k562_rand, order, palette, labels, fontsize, "K562", axarr[2], None, 
              "K562 MPRA activity", False, False, False, None)
plt.tight_layout()
f.savefig("Fig1_All_Biotypes_Random_Sel.pdf", bbox_inches="tight", dpi="figure")


# In[63]:


pool1_hela_exp = pool1_hela_df[pool1_hela_df["oligo_reg"].isin(exp_match_ids["oligo_reg"])]
pool1_hepg2_exp = pool1_hepg2_df[pool1_hepg2_df["oligo_reg"].isin(exp_match_ids["oligo_reg"])]
pool1_k562_exp = pool1_k562_df[pool1_k562_df["oligo_reg"].isin(exp_match_ids["oligo_reg"])]


# In[64]:


f, axarr = plt.subplots(3, sharex=True, sharey=False, figsize=(4, 8))
promtype_plot(pool1_hela_exp, order, palette, labels, fontsize, "HeLa", axarr[0], None, 
              "HeLa MPRA activity", False, False, False, None)
promtype_plot(pool1_hepg2_exp, order, palette, labels, fontsize, "HepG2", axarr[1], None, 
              "HepG2 MPRA activity", False, False, False, None)
promtype_plot(pool1_k562_exp, order, palette, labels, fontsize, "K562", axarr[2], None, 
              "K562 MPRA activity", False, False, False, None)
plt.tight_layout()
f.savefig("Fig1_All_Biotypes_ExpMatch_Sel.pdf", bbox_inches="tight", dpi="figure")


# the rest of the analysis only uses pool 1 (the TSS pool), as it looks at patterns in expression differences between TSS classes

# ## 6. barplots: find % of sequences active across cell types

# In[65]:


pool1_hela_df["cell"] = "HeLa"
pool1_hepg2_df["cell"] = "HepG2"
pool1_k562_df["cell"] = "K562"

all_df = pool1_hela_df[["unique_id", "better_type", "cell", "PromType2", "combined_class", "overall_mean"]].append(pool1_hepg2_df[["unique_id", "better_type", "cell", "PromType2", "combined_class", "overall_mean"]]).append(pool1_k562_df[["unique_id", "better_type", "cell", "PromType2", "combined_class", "overall_mean"]])


# In[66]:


df = all_df[all_df["better_type"] == "WILDTYPE"]
activ_grp = df.groupby("unique_id")["cell", "combined_class"].agg(lambda x: list(x)).reset_index()
activ_grp = activ_grp.merge(annot, left_on="unique_id", right_on="seqID", how="left").drop("seqID", axis=1)
activ_grp = activ_grp[(activ_grp["PromType2"].isin(TSS_CLASS_ORDER)) & 
                      ~(activ_grp["unique_id"].str.contains("SCRAMBLED"))]
activ_grp.sample(10)


# In[67]:


activ_grp["active_in_only_one"] = activ_grp.apply(active_in_only_one, axis=1)
activ_grp["active_in_only_two"] = activ_grp.apply(active_in_only_two, axis=1)
activ_grp["active_in_only_three"] = activ_grp.apply(active_in_only_three, axis=1)
activ_grp.sample(5)


# In[68]:


activ_counts_1 = activ_grp.groupby(["PromType2", "active_in_only_one"])["unique_id"].agg("count").reset_index()
activ_pcts_1 = activ_counts_1.groupby("PromType2")["unique_id"].apply(lambda x: 100 * x / float(x.sum()))
activ_counts_1["percent"] = activ_pcts_1

activ_counts_2 = activ_grp.groupby(["PromType2", "active_in_only_two"])["unique_id"].agg("count").reset_index()
activ_pcts_2 = activ_counts_2.groupby("PromType2")["unique_id"].apply(lambda x: 100 * x / float(x.sum()))
activ_counts_2["percent"] = activ_pcts_2

activ_counts_3 = activ_grp.groupby(["PromType2", "active_in_only_three"])["unique_id"].agg("count").reset_index()
activ_pcts_3 = activ_counts_3.groupby("PromType2")["unique_id"].apply(lambda x: 100 * x / float(x.sum()))
activ_counts_3["percent"] = activ_pcts_3

activ_counts_1 = activ_counts_1[activ_counts_1["active_in_only_one"]]
activ_counts_2 = activ_counts_2[activ_counts_2["active_in_only_two"]]
activ_counts_3 = activ_counts_3[activ_counts_3["active_in_only_three"]]

activ_counts = activ_counts_1.merge(activ_counts_2, on="PromType2").merge(activ_counts_3, on="PromType2")
activ_counts.drop(["active_in_only_one", "unique_id_x", "active_in_only_two", "unique_id_y", 
                   "active_in_only_three", "unique_id"],
                  axis=1, inplace=True)
activ_counts.columns = ["PromType2", "active_in_only_one", "active_in_only_two", "active_in_only_three"]
activ_counts = pd.melt(activ_counts, id_vars="PromType2")
activ_counts.head()


# In[69]:


df = activ_counts[activ_counts["PromType2"] != "antisense"]
df["PromType2"] = pd.Categorical(df["PromType2"], TSS_CLASS_ORDER)
df.sort_values(by="PromType2")

plt.figure(figsize=(3.56, 2.3))
ax = sns.barplot(data=df, x="variable", y="value", hue="PromType2", ci=None, palette=TSS_CLASS_PALETTE)
ax.set_xticklabels(["active in 1", "active in 2", "active in 3"], rotation=30)

plt.legend(bbox_to_anchor=(1.35, 1))
plt.ylim((0, 50))
plt.ylabel("percent of sequences", size=fontsize)
plt.xlabel("")
plt.title("% of elements active in # of cell types")


# In[70]:


colors = []
for c in TSS_CLASS_ORDER:
    colors.append(TSS_CLASS_PALETTE[c])
colors


# In[71]:


# better plot showing tissue sp
df = activ_counts[activ_counts["PromType2"] != "antisense"]
df["PromType2"] = pd.Categorical(df["PromType2"], TSS_CLASS_ORDER)
df.sort_values(by="PromType2")

plt.figure(figsize=(3,2.3))
ax = sns.barplot(data=df[df["variable"]!="active_in_only_two"], x="PromType2", y="value", 
                 ci=None, hue="variable", linewidth=1.5)
ax.set_xticklabels(["eRNAs", "lincRNAs", "div. lncRNAs", "mRNAs", "div. mRNAs"], rotation=30)

colors = colors*2
for i, p in enumerate(ax.patches):
    if i < 5:
        p.set_facecolor(colors[i])
    else:
        p.set_facecolor("white")
        p.set_edgecolor(colors[i])
        p.set_alpha(1)
        p.set_hatch("///")

ax.legend().set_visible(False)
plt.ylim((0, 50))
plt.ylabel("% of sequences that are active\nin 1 and 3 cell types", fontsize=fontsize)
plt.xlabel("")
plt.savefig("Fig_1F.pdf", bbox_inches="tight", dpi="figure")


# ## 7. kdeplot: compare to CAGE

# In[72]:


hepg2_activ = pool1_hepg2_df[["unique_id", "element", "better_type", "overall_mean", "PromType2"]]
hela_activ = pool1_hela_df[["unique_id", "element", "better_type", "overall_mean"]]
k562_activ = pool1_k562_df[["unique_id", "element", "better_type", "overall_mean"]]

all_activ = hepg2_activ.merge(hela_activ, on=["unique_id", "element", "better_type"], how="left").merge(k562_activ, on=["unique_id", "element", "better_type"], how="left")
all_activ.columns = ["unique_id", "element", "better_type", "HepG2", "PromType2", "HeLa", "K562"]
all_activ = all_activ[["unique_id", "element", "better_type", "PromType2", "HepG2", "HeLa", "K562"]]
all_activ = all_activ[(all_activ["PromType2"].isin(TSS_CLASS_ORDER)) & 
                      ~(all_activ["unique_id"].str.contains("SCRAMBLED")) &
                      (all_activ["better_type"] == "WILDTYPE")]
all_activ.sample(5)


# In[73]:


all_activ["combined_class"] = ""
all_activ = all_activ.merge(pool1_hela_elem_norm[["unique_id", "element", "combined_class"]], on=["unique_id", "element"], how="left", suffixes=("", "_HeLa")).merge(pool1_hepg2_elem_norm[["unique_id", "element", "combined_class"]], on=["unique_id", "element"], how="left", suffixes=("", "_HepG2")).merge(pool1_k562_elem_norm[["unique_id", "element", "combined_class"]], on=["unique_id", "element"], how="left", suffixes=("", "_K562"))
all_activ.drop("combined_class", axis=1, inplace=True)
all_activ.head()


# In[74]:


all_activ["oligo_reg"] = all_activ.unique_id.str.split("__", expand=True)[2]
all_activ.sample(5)


# In[75]:


id_map = id_map[["oligo_reg", "K562_rep1", "K562_rep2", "K562_rep3", "HeLa_rep1", "HeLa_rep2", "HeLa_rep3", 
                 "HepG2_rep1", "HepG2_rep2", "HepG2_rep3"]]
all_activ = all_activ.merge(id_map, on="oligo_reg")
all_activ.sample(5)


# In[76]:


all_activ["K562_av"] = all_activ[["K562_rep1", "K562_rep2", "K562_rep3"]].mean(axis=1)
all_activ["HeLa_av"] = all_activ[["HeLa_rep1", "HeLa_rep2", "HeLa_rep3"]].mean(axis=1)
all_activ["HepG2_av"] = all_activ[["HepG2_rep1", "HepG2_rep2", "HepG2_rep3"]].mean(axis=1)

all_activ["K562_log_av"] = np.log(all_activ["K562_av"]+1)
all_activ["HeLa_log_av"] = np.log(all_activ["HeLa_av"]+1)
all_activ["HepG2_log_av"] = np.log(all_activ["HepG2_av"]+1)


# In[77]:


all_activ = all_activ[(~all_activ["unique_id"].str.contains("SNP_INDIV")) & 
                      (~all_activ["unique_id"].str.contains("SNP_PLUS_HAPLO")) & 
                      (~all_activ["unique_id"].str.contains("FLIPPED"))]
all_activ.sample(5)


# In[78]:


# first scale mpra ranges to be positive
all_activ["hepg2_scaled"] = scale_range(all_activ["HepG2"], 0, 100)
all_activ["hela_scaled"] = scale_range(all_activ["HeLa"], 0, 100)
all_activ["k562_scaled"] = scale_range(all_activ["K562"], 0, 100)


# In[79]:


cage_ts = calculate_tissue_specificity(all_activ[["HepG2_log_av", "K562_log_av", "HeLa_log_av"]])
all_activ["cage_activ"] = all_activ[["HepG2_log_av", "K562_log_av", "HeLa_log_av"]].mean(axis=1)
all_activ["cage_ts"] = cage_ts

mpra_ts = calculate_tissue_specificity(all_activ[["hepg2_scaled", "k562_scaled", "hela_scaled"]])
all_activ["mpra_activ"] = all_activ[["HepG2", "K562", "HeLa"]].mean(axis=1)
all_activ["mpra_ts"] = mpra_ts
all_activ.head()


# In[80]:


cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[81]:


no_nan = all_activ[(~pd.isnull(all_activ["mpra_ts"])) & (~pd.isnull(all_activ["cage_ts"]))]
g = sns.jointplot(data=no_nan, x="cage_ts", y="mpra_ts", kind="kde", shade_lowest=False, size=2.3, space=0,
                  stat_func=None, cmap=cmap, color="darkslategrey")
g.ax_joint.axhline(y=0.2, color="black", linewidth=1, linestyle="dashed")
g.ax_joint.axvline(x=0.5, color="black", linewidth=1, linestyle="dashed")
g.set_axis_labels("CAGE cell-type specificity", "MPRA cell-type specificity")
r, p = stats.spearmanr(no_nan["cage_ts"], no_nan["mpra_ts"])
g.ax_joint.annotate("r = {:.2f}\np = {:.2e}".format(r, Decimal(p)), xy=(.1, .8), xycoords=ax.transAxes, 
                    fontsize=5)
g.savefig("Fig_1E.pdf", bbox_inches="tight", dpi="figure")


# In[82]:


no_nan = all_activ[(~pd.isnull(all_activ["mpra_ts"])) & (~pd.isnull(all_activ["cage_ts"]))]
g = sns.jointplot(data=no_nan, x="cage_ts", y="mpra_ts", kind="kde", shade_lowest=False, size=2.3, space=0,
                  stat_func=None, cmap=cmap, color="darkslategrey")
g.set_axis_labels("endogenous cell-type specificity", "MPRA cell-type specificity")
r, p = stats.spearmanr(no_nan["cage_ts"], no_nan["mpra_ts"])
g.ax_joint.annotate("r = {:.2f}\np = {:.2e}".format(r, Decimal(p)), xy=(.1, .75), xycoords=ax.transAxes, 
                    fontsize=5)
g.savefig("cage_mpra_corr.for_talk.pdf", bbox_inches="tight", dpi="figure")


# In[83]:


def cage_v_mpra_ts(row):
    if row["cage_ts"] > 0.5 and row["mpra_ts"] > 0.2:
        return "ts in both"
    elif row["cage_ts"] > 0.5 and row["mpra_ts"] <= 0.2:
        return "ts in cage, not mpra"
    elif row["cage_ts"] <= 0.5 and row["mpra_ts"] > 0.2:
        return "ts in mpra, not cage"
    else:
        return "not ts in both"
    
no_nan["ts_status"] = no_nan.apply(cage_v_mpra_ts, axis=1)
no_nan.ts_status.value_counts()


# In[84]:


tot = 692+402+310+236
upper_left = 310
upper_right = 402
lower_left = 692
lower_right = 236
print("upper left: %s" % (upper_left/tot))
print("upper right: %s" % (upper_right/tot))
print("lower left: %s" % (lower_left/tot))
print("lower right: %s" % (lower_right/tot))


# In[85]:


(692+402)/(692+402+310+236)


# In[86]:


no_nan = all_activ[(~pd.isnull(all_activ["mpra_activ"])) & (~pd.isnull(all_activ["cage_activ"]))]
g = sns.jointplot(data=no_nan, x="cage_activ", y="mpra_activ", kind="kde", shade_lowest=False, size=2.3, space=0,
                  stat_func=None, xlim=(-0.75, 1.75), ylim=(-3.5, 3), cmap=cmap, color="darkslategray")
g.set_axis_labels("mean CAGE expression", "mean MPRA activity")
r, p = stats.spearmanr(no_nan["cage_activ"], no_nan["mpra_activ"])
g.ax_joint.annotate("r = {:.2f}\np = {:.2e}".format(r, Decimal(p)), xy=(.1, .8), xycoords=ax.transAxes, 
                    fontsize=5)


# In[87]:


# write file with tissue-specificities for later use
final = all_activ[["unique_id", "PromType2", "cage_activ", "cage_ts", "mpra_activ", "mpra_ts"]]
final.to_csv("../../data/02__activs/POOL1__pMPRA1__CAGE_vs_MPRA_activs.txt", sep="\t", index=False)


# In[88]:


# also write file with tss types
sel_map_expr.to_csv("../../misc/00__tss_properties/CAGE_expr_properties.txt", sep="\t", index=False)


# In[ ]:




