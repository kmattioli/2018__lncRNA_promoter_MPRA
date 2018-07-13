
# coding: utf-8

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

from os import walk
from scipy.stats import spearmanr

# import utils
sys.path.append("../../utils")
from plotting_utils import *
from misc_utils import *
from norm_utils import *

get_ipython().magic('matplotlib inline')


# ## variables

# In[2]:


cage_f = "../../misc/03__rna_seq_expr/hg19.cage_peak_phase1and2combined_ann.txt.gz"


# In[3]:


fimo_f = "fimo_all_biotypes.txt"


# In[4]:


chip_f = "chip_all.txt"


# In[5]:


# files with tissue specificities calculated across all samples
tss_ts_f = "TSS.CAGE_grouped_exp.tissue_sp.txt"
enh_ts_f = "Enh.CAGE_grouped_exp.tissue_sp.txt"


# In[6]:


# files with CAGE expression in HepG2, HeLa, K562 only
tss_cell_line_expr_f = "../../misc/03__rna_seq_expr/hg19.cage_peak_tpm_ann.mpra_cell_line_replicates.tsv"
enh_cell_line_expr_f = "../../misc/03__rna_seq_expr/hg19.enhancers_tpm_ann.mpra_cell_line_replicates.tsv"


# ## 1. import data

# In[7]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo.head()


# In[8]:


chip = pd.read_table(chip_f, sep="\t")
chip.head()


# In[9]:


cage = pd.read_table(cage_f, sep="\t", skiprows=7, header=0)
cage.head()


# In[10]:


tss_ts = pd.read_table(tss_ts_f, sep="\t")
tss_ts.head()


# In[11]:


enh_ts = pd.read_table(enh_ts_f, sep="\t")
enh_ts.head()


# ## 2. find unique TF names

# In[19]:


fimo_tfs = list(fimo["#pattern name"].unique())
len(fimo_tfs)


# In[20]:


chip_tfs = list(chip["chip_id"].unique())
len(chip_tfs)


# In[21]:


fimo_tfs.extend(chip_tfs)
all_tfs = list(set(fimo_tfs))
len(all_tfs)


# In[22]:


all_tfs = [x.upper() for x in all_tfs]
all_tfs[0:5]


# In[23]:


# remove fusion proteins and vars from list (ones with ::)
all_tfs = [x for x in all_tfs if "::" not in x and "(VAR" not in x]
len(all_tfs)


# ## 3. determine how many TF names are missing from CAGE file

# In[51]:


def get_gene_names(row):
    sep_proms = row.short_description.split("@")
    sep_proms = [x.split(",") for x in sep_proms]
    genes = []
    for x in sep_proms:
        for s in x:
            if s == "+" or s == "-":
                continue
            elif s.startswith("p"):
                continue
            else:
                if s.startswith("chr"):
                    genes.append(s)
                else:
                    genes.append(s.upper())
    genes = list(set(genes))
    genes = ",".join(map(str, genes)) 
    return genes


# In[52]:


def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df


# In[53]:


cage["genes"] = cage.apply(get_gene_names, axis=1)
cage.sample(5)


# In[54]:


cage_split = tidy_split(cage, column="genes", sep=",")
cage_split.head()


# In[57]:


cage_split[cage_split["genes"].astype(str).str.contains("MEF2B")]


# In[58]:


manual_fixes = {"SIN3AK20": "SIN3A", "KAP1": "TRIM28", "SREBP1": "SREBF1", "MIX-A": "MIXL1", 
                "RPC155": "POLR3A", "ZBTB18": "ZNF238"}


# In[59]:


cage_genes = list(cage_split["genes"].unique())


# In[64]:


tf_cage_map = {}
counter = 0
for tf in all_tfs:
    if tf in cage_genes:
        sub = cage_split[cage_split["genes"] == tf]
        peaks = list(sub["00Annotation"].unique())
    elif tf in manual_fixes:
        name = manual_fixes[tf]
        sub = cage_split[cage_split["short_description"] == name]
        peaks = list(sub["00Annotation"].unique())
    else:
        peaks = ["none"]
        counter += 1
    tf_cage_map[tf] =  peaks


# In[65]:


counter


# only 12/491 (non-var, non-fusion TFs) cannot be mapped this way

# ## 4. map peaks to tissue specificity values

# In[66]:


tss_ts.head()


# In[95]:


sample_cols = [x for x in tss_ts.columns if "Group_" in x]


# In[96]:


tf_ts_map = {}
for tf in tf_cage_map:
    peaks = tf_cage_map[tf]
    tss_ts_sub = tss_ts[tss_ts["00Annotation"].isin(peaks)]
    if len(tss_ts_sub) > 0:
        tss_ts_sub_sum = tss_ts_sub[sample_cols].sum(axis=0)
        sub_array = np.zeros((1, len(sample_cols)))
        sub_array[0,:] = tss_ts_sub_sum
        sub_df = pd.DataFrame(data=sub_array)
        specificity = calculate_tissue_specificity(sub_df)
        tf_ts_map[tf] = specificity


# In[97]:


tf_ts_map = pd.DataFrame.from_dict(tf_ts_map, orient="index").reset_index()
tf_ts_map.columns = ["tf", "tissue_sp_all"]
tf_ts_map.sort_values(by="tissue_sp_all").head()


# ## 5. calculate tissue-specificity based on 3 MPRA cell lines

# In[98]:


K562_group = "Group_17"
HepG2_group = "Group_513"
HeLa_group = "Group_512"
sample_3_cols = [K562_group, HepG2_group, HeLa_group]
sample_3_cols


# In[99]:


tf_ts_map_3 = {}
for tf in tf_cage_map:
    peaks = tf_cage_map[tf]
    tss_ts_sub = tss_ts[tss_ts["00Annotation"].isin(peaks)]
    if len(tss_ts_sub) > 0:
        tss_ts_sub_sum = tss_ts_sub[sample_3_cols].sum(axis=0)
        sub_array = np.zeros((1, len(sample_3_cols)))
        sub_array[0,:] = tss_ts_sub_sum
        sub_df = pd.DataFrame(data=sub_array)
        specificity = calculate_tissue_specificity(sub_df)
        tf_ts_map_3[tf] = specificity


# In[100]:


tf_ts_map_3 = pd.DataFrame.from_dict(tf_ts_map_3, orient="index").reset_index()
tf_ts_map_3.columns = ["tf", "tissue_sp_3"]
tf_ts_map_3.sort_values(by="tissue_sp_3").head()


# In[101]:


tf_ts_map = tf_ts_map.merge(tf_ts_map_3, on="tf", how="left")
tf_ts_map.sample(5)


# ## 6. calculate tissue sp based on 3 for all TSS and all enh

# In[102]:


specificities = calculate_tissue_specificity(tss_ts[sample_3_cols])
tss_ts["tissue_sp_3"] = specificities
tss_ts.sample(5)


# In[103]:


specificities = calculate_tissue_specificity(enh_ts[sample_3_cols])
enh_ts["tissue_sp_3"] = specificities
enh_ts.sample(5)


# ## 7. write files

# In[104]:


tf_ts_map.to_csv("TF_tissue_specificities.from_CAGE.txt", sep="\t", index=False)
len(tf_ts_map)


# In[105]:


tss_ts.to_csv("TSS.CAGE_grouped_exp.tissue_sp.txt", sep="\t", index=False)


# In[106]:


enh_ts.to_csv("Enh.CAGE_grouped_exp.tissue_sp.txt", sep="\t", index=False)


# In[ ]:




