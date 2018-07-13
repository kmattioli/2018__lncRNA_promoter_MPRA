
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


tss_cage_exp_f = "/n/rinn_data2/users/kaia/fantom5/hg19.cage_peak_phase1and2combined_tpm_ann.osc.txt.gz"
enh_cage_exp_f = "/n/rinn_data2/users/kaia/fantom5/human_permissive_enhancers_phase_1_and_2_expression_tpm_matrix.txt"


# In[3]:


# grouped by hand (ugghh)
# group different biological/technical replicates together
# omit timecourses (differentiation or infection), fractionations, and ambiguous labels
human_cage_samples_f = "hg19.CAGE_samples.grouped.txt"


# ## 1. find CAGE samples to use for tissue-spec calculations

# In[4]:


human_cage_samples = pd.read_table(human_cage_samples_f, sep="\t")
print(len(human_cage_samples))
human_cage_samples.head()


# In[5]:


human_cage_samples = human_cage_samples[human_cage_samples["group"] != "omit"]
len(human_cage_samples)


# In[6]:


cage_samples_grp = human_cage_samples.groupby("group")["sample"].apply(list).reset_index()
cage_samples_grp.drop("group", axis=1, inplace=True)
cage_samples_grp.reset_index(inplace=True)
print(len(cage_samples_grp))
cage_samples_grp.sample(5)


# ## 2. find average expression for each group of samples

# In[7]:


tss_cage_exp = pd.read_table(tss_cage_exp_f, sep="\t", skiprows=1837, header=0)
tss_cage_exp.head()


# In[20]:


enh_cage_exp = pd.read_table(enh_cage_exp_f, sep="\t")
enh_cage_exp.head()


# In[8]:


tss_cage_exp = tss_cage_exp[~tss_cage_exp["00Annotation"].isin(["01STAT:MAPPED", "02STAT:NORM_FACTOR"])]


# In[9]:


grouped_samples = list(cage_samples_grp["sample"])
for i, group in enumerate(grouped_samples):
    print("i: %s, group: %s" % (i, group))
    cols_to_extract = []
    for sample in group:
        col = [x for x in tss_cage_exp.columns if sample in x]
        cols_to_extract.extend(col) 
    if len(group) == 1:
        tss_cage_exp["Group_%s" % i] = tss_cage_exp[cols_to_extract]
    else:
        mean_exp = tss_cage_exp[cols_to_extract].mean(axis=1)
        tss_cage_exp["Group_%s" % i] = mean_exp
    tss_cage_exp.drop(cols_to_extract, axis=1, inplace=True)
tss_cage_exp.head()


# In[21]:


for i, group in enumerate(grouped_samples):
    print("i: %s, group: %s" % (i, group))
    cols_to_extract = []
    for sample in group:
        col = [x for x in enh_cage_exp.columns if sample in x]
        cols_to_extract.extend(col) 
    if len(group) == 1:
        enh_cage_exp["Group_%s" % i] = enh_cage_exp[cols_to_extract]
    else:
        mean_exp = enh_cage_exp[cols_to_extract].mean(axis=1)
        enh_cage_exp["Group_%s" % i] = mean_exp
    enh_cage_exp.drop(cols_to_extract, axis=1, inplace=True)
enh_cage_exp.head()


# ## 3. calculate tissue specificity

# In[11]:


cols = ["00Annotation", "short_description"]
sample_cols = [x for x in tss_cage_exp.columns if "Group_" in x]
cols.extend(sample_cols)
tss_cage_grp = tss_cage_exp[cols]
tss_cage_grp.sample(5)


# In[23]:


cols = ["Id"]
cols.extend(sample_cols)
enh_cage_grp = enh_cage_exp[cols]
enh_cage_grp.sample(5)


# In[12]:


specificities = calculate_tissue_specificity(tss_cage_grp[sample_cols])
tss_cage_grp["tissue_sp_all"] = specificities
tss_cage_grp.sample(5)


# In[24]:


specificities = calculate_tissue_specificity(enh_cage_grp[sample_cols])
enh_cage_grp["tissue_sp_all"] = specificities
enh_cage_grp.sample(5)


# In[14]:


print("TSSs total: %s" % (len(tss_cage_grp)))
tss_cage_grp_nonan = tss_cage_grp[~pd.isnull(tss_cage_grp["tissue_sp_all"])]
print("TSSs no NANs: %s" % (len(tss_cage_grp_nonan)))


# In[25]:


print("Enhancers total: %s" % (len(enh_cage_grp)))
enh_cage_grp_nonan = enh_cage_grp[~pd.isnull(enh_cage_grp["tissue_sp_all"])]
print("Enhancers no NANs: %s" % (len(enh_cage_grp_nonan)))


# In[15]:


sns.distplot(tss_cage_grp_nonan["tissue_sp_all"])


# In[26]:


sns.distplot(enh_cage_grp_nonan["tissue_sp_all"])


# In[16]:


tss_cage_grp_nonan[tss_cage_grp_nonan["short_description"].str.contains("POU5F1")]


# In[17]:


tss_cage_grp_nonan[tss_cage_grp_nonan["short_description"].str.contains("POLR2A")]


# In[18]:


tss_cage_grp_nonan.sort_values(by="tissue_sp_all").head()


# ## 4. write files

# In[27]:


cage_samples_grp.to_csv("CAGE_samples_grouped.txt", sep="\t", index=False)
tss_cage_grp_nonan.to_csv("TSS.CAGE_grouped_exp.tissue_sp.txt", sep="\t", index=False)
enh_cage_grp_nonan.to_csv("Enh.CAGE_grouped_exp.tissue_sp.txt", sep="\t", index=False)


# In[ ]:




