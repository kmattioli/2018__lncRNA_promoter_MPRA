
# coding: utf-8

# # 01__preprocess
# # turn files of counts into large dataframe of counts
# in this notebook, I load in the barcode counts that we got from the .fastq files and put everything in one nice dataframe. additionally, I filter the barcodes to only include those that have >= 5 reads in a given sample.
# 
# - throughout these notebooks, "pool1" refers to the TSS pool
# - pool1 was performed in HepG2, HeLa, and K562
# 
# ------
# 
# figures in this notebook:
# - **Fig S3A**: heatmap of barcode count correlation between replicates

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import sys

# import utils
sys.path.append("../../utils")
from plotting_utils import *

get_ipython().magic('matplotlib inline')


# In[2]:


sns.set(**NOTEBOOK_PRESET)
fontsize = NOTEBOOK_FONTSIZE


# ## functions

# In[3]:


def import_dna(counts_dir, dna_f):
    """
    function that will import the DNA counts for a given experiment
    ----
    counts_dir: string defining where the counts files are stored
    dna_f: list of strings defining where the DNA counts files are
    
    returns a dataframe with barcodes + counts
    """
    dna_dfs = []
    for i in range(len(dna_f)):
        f = dna_f[i]
        cols = ["barcode", "dna_%s" % (i+1)]
        tmp = pd.read_table("%s/%s" % (counts_dir, f), sep="\t", header=None)
        tmp.columns = cols
        dna_dfs.append(tmp)
    if len(dna_dfs) > 1:
        dna = reduce(lambda x, y: pd.merge(x, y, on = "barcode"), dna_dfs)
    else:
        dna = dna_dfs[0]
    return dna


# In[4]:


def import_rna(counts_dir, rna_f, dna):
    """
    function that will import the RNA counts for a given experiment
    ----
    counts_dir: string defining where the counts files are stored
    rna_f: list of strings defining where the RNA counts files are
    dna: dataframe w/ DNA counts
    
    returns a dataframe with barcodes + counts, incl. RNA
    """
    data = dna.copy()
    data_cols = list(dna.columns)
    for f in rna_f:
        rep = re.findall(r'\d+', f.split("__")[3])[0]
        tmp = pd.read_table("%s/%s" % (counts_dir, f), sep="\t", header=None)
        tmp.columns = ["barcode", "rna_%s" % rep]
        data_cols.append("rna_%s" % rep)
        data = data.merge(tmp, on="barcode", how="outer")
    return data, data_cols


# ## variables

# In[5]:


counts_dir = "../../data/01__counts"


# In[6]:


barcode_dna_read_threshold = 5
barcode_rna_read_threshold = 5


# ### DNA files

# In[7]:


pool1_pMPRA1_dna_f = ["POOL1__pMPRA1__COUNTS.txt"]


# ### RNA files

# In[8]:


pool1_pMPRA1_HeLa_rna_f = ["POOL1__pMPRA1__HeLa__rep1__COUNTS.txt",
                           "POOL1__pMPRA1__HeLa__rep2__COUNTS.txt",
                           "POOL1__pMPRA1__HeLa__rep3__COUNTS.txt", 
                           "POOL1__pMPRA1__HeLa__rep4__COUNTS.txt"]
pool1_pMPRA1_HeLa_out_f = "POOL1__pMPRA1__HeLa__all_counts.txt"


# In[9]:


pool1_pMPRA1_HepG2_rna_f = ["POOL1__pMPRA1__HepG2__rep3__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep4__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep5__COUNTS.txt", 
                            "POOL1__pMPRA1__HepG2__rep6__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep7__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep8__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep9__COUNTS.txt", 
                            "POOL1__pMPRA1__HepG2__rep10__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep11__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep12__COUNTS.txt",
                            "POOL1__pMPRA1__HepG2__rep13__COUNTS.txt", 
                            "POOL1__pMPRA1__HepG2__rep14__COUNTS.txt"]
pool1_pMPRA1_HepG2_out_f = "POOL1__pMPRA1__HepG2__all_counts.txt"


# In[10]:


pool1_pMPRA1_K562_rna_f = ["POOL1__pMPRA1__K562__rep1__COUNTS.txt",
                           "POOL1__pMPRA1__K562__rep2__COUNTS.txt",
                           "POOL1__pMPRA1__K562__rep3__COUNTS.txt", 
                           "POOL1__pMPRA1__K562__rep4__COUNTS.txt"]
pool1_pMPRA1_K562_out_f = "POOL1__pMPRA1__K562__all_counts.txt"


# ### Index Files

# In[11]:


pool1_index_f = "../../data/00__index/tss_oligo_pool.index.txt"


# ## 1. import indexes

# In[12]:


pool1_index = pd.read_table(pool1_index_f, sep="\t")
pool1_index.head()


# ## 2. import dna

# In[13]:


pool1_pMPRA1_dna = import_dna(counts_dir, pool1_pMPRA1_dna_f)
pool1_pMPRA1_dna.head()


# ## 3. import rna

# In[14]:


pool1_pMPRA1_HeLa_data, pool1_pMPRA1_HeLa_cols = import_rna(counts_dir, pool1_pMPRA1_HeLa_rna_f, pool1_pMPRA1_dna)
pool1_pMPRA1_HeLa_data.head()


# In[15]:


pool1_pMPRA1_HepG2_data, pool1_pMPRA1_HepG2_cols = import_rna(counts_dir, pool1_pMPRA1_HepG2_rna_f, pool1_pMPRA1_dna)
pool1_pMPRA1_HepG2_data.head()


# In[16]:


pool1_pMPRA1_K562_data, pool1_pMPRA1_K562_cols = import_rna(counts_dir, pool1_pMPRA1_K562_rna_f, pool1_pMPRA1_dna)
pool1_pMPRA1_K562_data.head()


# ## 4. filter barcodes

# In[17]:


# print HeLa df
pool1_pMPRA1_HeLa_data.head()


# In[18]:


# fill NA values with 0
pool1_pMPRA1_HeLa_data = pool1_pMPRA1_HeLa_data.fillna(0)
pool1_pMPRA1_HepG2_data = pool1_pMPRA1_HepG2_data.fillna(0)
pool1_pMPRA1_K562_data = pool1_pMPRA1_K562_data.fillna(0)
pool1_pMPRA1_HeLa_data.head()


# In[20]:


# subset dataframes to rows where DNA counts are greater than the threshold
pool1_pMPRA1_HeLa_data_filt = pool1_pMPRA1_HeLa_data[pool1_pMPRA1_HeLa_data["dna_1"] >= barcode_dna_read_threshold]
pool1_pMPRA1_HepG2_data_filt = pool1_pMPRA1_HepG2_data[pool1_pMPRA1_HepG2_data["dna_1"] >= barcode_dna_read_threshold]
pool1_pMPRA1_K562_data_filt = pool1_pMPRA1_K562_data[pool1_pMPRA1_K562_data["dna_1"] >= barcode_dna_read_threshold]
pool1_pMPRA1_HeLa_data_filt.head()


# In[21]:


# grab the RNA replicate column names
pool1_pMPRA1_HeLa_reps = [x for x in pool1_pMPRA1_HeLa_data_filt.columns if "rna_" in x]
pool1_pMPRA1_HepG2_reps = [x for x in pool1_pMPRA1_HepG2_data_filt.columns if "rna_" in x]
pool1_pMPRA1_K562_reps = [x for x in pool1_pMPRA1_K562_data_filt.columns if "rna_" in x]
pool1_pMPRA1_HeLa_reps


# In[22]:


# if a replicate column has fewer than the RNA counst threshold, fill it with NA
pool1_pMPRA1_HeLa_data_filt[pool1_pMPRA1_HeLa_reps] = pool1_pMPRA1_HeLa_data_filt[pool1_pMPRA1_HeLa_data_filt > barcode_rna_read_threshold][pool1_pMPRA1_HeLa_reps]
pool1_pMPRA1_HepG2_data_filt[pool1_pMPRA1_HepG2_reps] = pool1_pMPRA1_HepG2_data_filt[pool1_pMPRA1_HepG2_data_filt > barcode_rna_read_threshold][pool1_pMPRA1_HepG2_reps]
pool1_pMPRA1_K562_data_filt[pool1_pMPRA1_K562_reps] = pool1_pMPRA1_K562_data_filt[pool1_pMPRA1_K562_data_filt > barcode_rna_read_threshold][pool1_pMPRA1_K562_reps]
pool1_pMPRA1_HeLa_data_filt.head()


# In[23]:


# count the # of barcodes that meet the DNA/RNA threshold requirements

all_names = ["pool1_pMPRA1_HeLa", "pool1_pMPRA1_HepG2", "pool1_pMPRA1_K562"]
all_dfs = [pool1_pMPRA1_HeLa_data_filt, pool1_pMPRA1_HepG2_data_filt, pool1_pMPRA1_K562_data_filt]
all_cols = [pool1_pMPRA1_HeLa_cols, pool1_pMPRA1_HepG2_cols, pool1_pMPRA1_K562_cols]

print("FILTERING RESULTS:")
for n, df, cs in zip(all_names, all_dfs, all_cols):
    index_len = len(pool1_index)
    dna_barc_len = len(df)
    dna_barc_perc = (float(dna_barc_len)/index_len)*100
    
    print("%s: from %s barcodes to %s at DNA level (%s%%)" % (n, index_len, dna_barc_len, dna_barc_perc))
    
    reps = [x for x in cs if "rna_" in x]
    for r in reps:
        rep = r.split("_")[1]
        rna_barc_len = sum(~pd.isnull(df[r]))
        rna_barc_perc = (float(rna_barc_len)/index_len)*100
        
        print("\trep %s: %s barcodes at RNA level (%s%%)" % (rep, rna_barc_len, rna_barc_perc))
    print("")


# ## 5. heatmap comparing barcode counts

# In[24]:


# start lists each containing "barcode"
pool1_pMPRA1_HeLa_cols = ["barcode"]
pool1_pMPRA1_HepG2_cols = ["barcode"]
pool1_pMPRA1_K562_cols = ["barcode"]


# In[25]:


# extend lists with more detailed column names
pool1_pMPRA1_HeLa_cols.extend(["HeLa_%s" % x for x in pool1_pMPRA1_HeLa_reps])
pool1_pMPRA1_HepG2_cols.extend(["HepG2_%s" % x for x in pool1_pMPRA1_HepG2_reps])
pool1_pMPRA1_K562_cols.extend(["K562_%s" % x for x in pool1_pMPRA1_K562_reps])
pool1_pMPRA1_HeLa_cols


# In[26]:


# drop the DNA column as it's the same for all of them
pool1_pMPRA1_HeLa_data_filt.drop("dna_1", axis=1, inplace=True)
pool1_pMPRA1_HepG2_data_filt.drop("dna_1", axis=1, inplace=True)
pool1_pMPRA1_K562_data_filt.drop("dna_1", axis=1, inplace=True)
pool1_pMPRA1_HeLa_data_filt.head()


# In[27]:


# set new column names
pool1_pMPRA1_HeLa_data_filt.columns = pool1_pMPRA1_HeLa_cols
pool1_pMPRA1_HepG2_data_filt.columns = pool1_pMPRA1_HepG2_cols
pool1_pMPRA1_K562_data_filt.columns = pool1_pMPRA1_K562_cols


# In[29]:


# merge all dataframes and correlate them
all_pool1 = pool1_pMPRA1_HeLa_data_filt.merge(pool1_pMPRA1_HepG2_data_filt, on="barcode").merge(pool1_pMPRA1_K562_data_filt, on = "barcode")
all_pool1_corr = all_pool1.corr(method="pearson")
all_pool1_corr


# In[30]:


# make a heatmap

# set the colormap
cmap = sns.cubehelix_palette(as_cmap=True)

# create the heatmap object
cg = sns.clustermap(all_pool1_corr, figsize=(7.2,7.2), cmap=cmap, annot=False)

# rotate the labels so they don't overlap
_ = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

# add a title
plt.suptitle("pearson correlation of replicates\nall barcodes at counts > 5")

# save it
# cg.savefig("Fig_S3A.pdf", dpi="figure", transparent=True, bbox_inches="tight")


# In[31]:


all_pool1.head()


# In[ ]:




