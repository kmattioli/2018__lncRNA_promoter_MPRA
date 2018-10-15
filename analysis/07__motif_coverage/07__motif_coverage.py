
# coding: utf-8

# # 07__motif_coverage
# # analyzing how motif coverage correlates with MPRA properties and biotypes; clustering similar motifs
# 
# in this notebook, i look at how the coverage metrics (# bp covered and max coverage of motifs; done separately, see methods) look within biotypes *after* limiting to only those motifs which have been validated by a corresponding chip peak. i also make sure the results we see aren't due to redundancies in motif databases, so i cluster the motifs using MoSBAT (done separately using their webtool) and re-calculate the metrics.
# 
# ------
# 
# figures in this notebook:
# - **Fig 2D and 2E**: cumulative density plots of # bp covered and max motif coverage across biotypes
# - **Fig S7**: heatmap of clustered motifs, and more cumulative density plots (after clustering)

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
from random import shuffle
from scipy import stats
from scipy import signal
from scipy.spatial import distance
from scipy.cluster import hierarchy
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


mosbat_file = "../../misc/02__mosbat/results.from_mosbat.txt"


# In[4]:


all_tss_f = "../../data/00__index/0__all_tss/All.TSS.114bp.bed"
motif_cov_f = "../../misc/03__fimo/All.TSS.114bp.Motifs.txt"
motif_max_f = "../../misc/03__fimo/All.TSS.114bp.maxonly.txt"
chip_cov_f = "../../misc/03__fimo/All.TSS.114bp.Motifs.Intersect.Chip.ALL.txt"
cluster_cov_f = "../../misc/03__fimo/All.TSS.114bp.Cluster.ALL.txt"
cage_expr_f = "../../misc/other_files/All_TSS_and_enh.CAGE_grouped_exp.tissue_sp.txt"


# note: the reason why some IDs are not in the expression file is these are FANTOM CAT IDs that for some reason are not present in the FANTOM5 robust set. so, we exclude these.

# ## 1. import data

# In[5]:


corr = pd.read_table(mosbat_file, sep="\t")


# In[6]:


all_tss = pd.read_table(all_tss_f, sep="\t", header=None)
all_tss.columns = ["chr", "start", "end", "seqID", "score", "strand"]
all_tss = all_tss.drop_duplicates()
print(len(all_tss))
print(len(all_tss["seqID"].unique()))
all_tss.head()


# In[7]:


motif_cov = pd.read_table(motif_cov_f, sep="\t")
motif_cov = motif_cov.drop_duplicates()
motif_max = pd.read_table(motif_max_f, sep="\t", header=None)
motif_max = motif_max.drop_duplicates()
motif_max.columns = ["seqID", "MaxCov"]

# get promtype2
motif_cov["PromType2"] = motif_cov.seqID.str.split("__", expand=True)[0]


# merge w/ All_TSS to get 0s
motif_cov = all_tss[["seqID"]].merge(motif_cov, on="seqID", how="left")
motif_cov = motif_cov.merge(motif_max, on="seqID", how="left")
motif_cov.fillna(0, inplace=True)
print(len(motif_cov))

# get promtype2
motif_cov["PromType2"] = motif_cov.seqID.str.split("__", expand=True)[0]


motif_cov.sample(5)


# In[8]:


chip_cov = pd.read_table(chip_cov_f, sep="\t")
chip_cov = chip_cov.drop_duplicates()

# get promtype2
chip_cov["PromType2"] = chip_cov.seqID.str.split("__", expand=True)[0]

# merge w/ All_TSS to get 0s
chip_cov = all_tss[["seqID"]].merge(chip_cov, on="seqID", how="left")
chip_cov.fillna(0, inplace=True)
print(len(chip_cov))

# get promtype2
chip_cov["PromType2"] = chip_cov.seqID.str.split("__", expand=True)[0]


chip_cov.sample(5)


# In[9]:


def fix_enh_seqid(row):
    if "Enhancer" in row.seqID:
        if "NA" in row.seqID:
            new_id = row.seqID.split("__")[0] + "__" + row.seqID.split("__")[2]
            return new_id
        else:
            return row.seqID
    else:
        return row.seqID


# In[10]:


cluster_cov = pd.read_table(cluster_cov_f, sep="\t")
cluster_cov = cluster_cov.drop_duplicates()

# get promtype2
cluster_cov["PromType2"] = cluster_cov.seqID.str.split("__", expand=True)[0]
cluster_cov["seqID"] = cluster_cov.apply(fix_enh_seqid, axis=1)

# # merge w/ All_TSS to get 0s
cluster_cov = all_tss[["seqID"]].merge(cluster_cov, on="seqID", how="left")
cluster_cov.fillna(0, inplace=True)
print(len(cluster_cov))

# get promtype2
cluster_cov["PromType2"] = cluster_cov.seqID.str.split("__", expand=True)[0]

cluster_cov.sample(5)


# In[11]:


# filter to those that have at least 1 motif so distributions are not 0-skewed
motif_cov = motif_cov[motif_cov["numMotifs"] > 0]
print(len(motif_cov))

chip_cov = chip_cov[chip_cov["numMotifs"] > 0]
print(len(chip_cov))

cluster_cov = cluster_cov[cluster_cov["numMotifs"] > 0]
print(len(cluster_cov))


# In[12]:


cage_expr = pd.read_table(cage_expr_f, sep="\t")
cage_expr.head()


# In[13]:


def get_cage_id(row):
    split = row.seqID.split("__")
    if len(split) == 2:
        return split[1]
    else:
        return split[2]


# In[14]:


motif_cov["cage_id"] = motif_cov.apply(get_cage_id, axis=1)
motif_cov = motif_cov.merge(cage_expr, on="cage_id", how="left")
motif_cov.head()


# In[15]:


chip_cov["cage_id"] = chip_cov.apply(get_cage_id, axis=1)
chip_cov = chip_cov.merge(cage_expr, on="cage_id", how="left")
chip_cov.sample(10)


# In[16]:


cluster_cov["cage_id"] = cluster_cov.apply(get_cage_id, axis=1)
cluster_cov = cluster_cov.merge(cage_expr, on="cage_id", how="left")
cluster_cov.head()


# In[17]:


chip_cov_exp = chip_cov[~pd.isnull(chip_cov["av_exp"])]
motif_cov_exp = motif_cov[~pd.isnull(motif_cov["av_exp"])]
cluster_cov_exp = cluster_cov[~pd.isnull(cluster_cov["av_exp"])]


# In[18]:


motif_cov_exp.PromType2.value_counts()


# In[19]:


chip_cov_exp.PromType2.value_counts()


# In[20]:


cluster_cov_exp.PromType2.value_counts()


# ## 2. plot # bp covered & max cov across biotypes

# ### all motifs

# In[21]:


motif_cov["log_bp_covered"] = np.log(motif_cov["numBPcovered"]+1)
motif_cov["log_max_cov"] = np.log(motif_cov["MaxCov"]+1)

motif_cov_exp["log_bp_covered"] = np.log(motif_cov_exp["numBPcovered"]+1)
motif_cov_exp["log_max_cov"] = np.log(motif_cov_exp["MaxCov"]+1)


# In[22]:


enh_vals = motif_cov[motif_cov["PromType2"] == "Enhancer"]["log_bp_covered"]
linc_vals = motif_cov[motif_cov["PromType2"] == "intergenic"]["log_bp_covered"]
dlnc_vals = motif_cov[motif_cov["PromType2"] == "div_lnc"]["log_bp_covered"]
pc_vals = motif_cov[motif_cov["PromType2"] == "protein_coding"]["log_bp_covered"]
dpc_vals = motif_cov[motif_cov["PromType2"] == "div_pc"]["log_bp_covered"]

fig = plt.figure(figsize=(2.75, 2))
ax = sns.kdeplot(data=enh_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["Enhancer"], 
                 label="eRNAs (n=%s)" % len(enh_vals))
sns.kdeplot(data=linc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["intergenic"], 
            label="lincRNAs (n=%s)" % len(linc_vals), ax=ax)
sns.kdeplot(data=dlnc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_lnc"], 
            label="div. lncRNAs (n=%s)" % len(dlnc_vals), ax=ax)
sns.kdeplot(data=pc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs (n=%s)" % len(pc_vals), ax=ax)
sns.kdeplot(data=dpc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_pc"], 
            label="div. mRNAs (n=%s)" % len(dpc_vals), ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
ax.set_xlim((2, 5))


# In[23]:


# for each group, split into tissue-sp v dynamic v ubiquitous
idxs = list(range(0, 5))
promtypes = ["Enhancer", "intergenic", "protein_coding"]
names = ["eRNAs", "lincRNAs", "mRNAs"]
df = motif_cov_exp
col = "log_bp_covered"
xlabel = "log(# of bp covered)"
xlim = (2, 5)

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(7, 2))

for i, promtype, name in zip(idxs, promtypes, names):
    print("i: %s, promtype: %s, name: %s" % (i, promtype, name))
    ax = axarr[i]
    
    promtype_vals = df[df["PromType2"] == promtype]
    ts_vals = promtype_vals[promtype_vals["tss_type"] == "tissue-specific"][col]
    ub_vals = promtype_vals[promtype_vals["tss_type"] == "ubiquitous"][col]
    dy_vals = promtype_vals[promtype_vals["tss_type"] == "dynamic"][col]
    
    sns.kdeplot(data=ts_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dashed",
                label="tissue-specific\n(n=%s)" % len(ts_vals), ax=ax)
    sns.kdeplot(data=ub_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="solid",
                label="ubiquitous\n(n=%s)" % len(ub_vals), ax=ax)
    sns.kdeplot(data=dy_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dotted",
                label="dynamic\n(n=%s)" % len(dy_vals), ax=ax)
    
    ax.set_title("%s (%s total)" % (name, len(promtype_vals)))
    ax.set_xlabel(xlabel)
    ax.legend(loc="upper left")
    ax.set_xlim(xlim)
    
    if i == 0:
        ax.set_ylabel("cumulative density")


# In[24]:


fig = plt.figure(figsize=(2, 1.7))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
ax.set_xlim((2, 5))
ax.set_ylim((0, 1.02))
fig.savefig("num_bp_cov.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[25]:


enh_vals = motif_cov[motif_cov["PromType2"] == "Enhancer"]["log_max_cov"]
linc_vals = motif_cov[motif_cov["PromType2"] == "intergenic"]["log_max_cov"]
dlnc_vals = motif_cov[motif_cov["PromType2"] == "div_lnc"]["log_max_cov"]
pc_vals = motif_cov[motif_cov["PromType2"] == "protein_coding"]["log_max_cov"]
dpc_vals = motif_cov[motif_cov["PromType2"] == "div_pc"]["log_max_cov"]

fig = plt.figure(figsize=(2.75, 2))
ax = sns.kdeplot(data=enh_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["Enhancer"], 
                 label="eRNAs\n(n=%s)" % len(enh_vals))
sns.kdeplot(data=linc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["intergenic"], 
            label="lincRNAs\n(n=%s)" % len(linc_vals), ax=ax)
sns.kdeplot(data=dlnc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_lnc"], 
            label="div. lncRNAs\n(n=%s)" % len(dlnc_vals), ax=ax)
sns.kdeplot(data=pc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs\n(n=%s)" % len(pc_vals), ax=ax)
sns.kdeplot(data=dpc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_pc"], 
            label="div. mRNAs\n(n=%s)" % len(dpc_vals), ax=ax)
ax.set_xlabel("log(max coverage)")
ax.set_ylabel("cumulative density")
ax.legend(loc="upper left")
fig.savefig("max_cov.all_biotypes.for_poster.pdf", dpi="figure", bbox_inches="tight")


# In[26]:


# for each group, split into tissue-sp v dynamic v ubiquitous
idxs = list(range(0, 5))
promtypes = ["Enhancer", "intergenic", "protein_coding"]
names = ["eRNAs", "lincRNAs", "mRNAs"]
df = motif_cov_exp
col = "log_max_cov"
xlabel = "log(max coverage)"
xlim = (0, 5.5)

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(7, 2))

for i, promtype, name in zip(idxs, promtypes, names):
    print("i: %s, promtype: %s, name: %s" % (i, promtype, name))
    ax = axarr[i]
    
    promtype_vals = df[df["PromType2"] == promtype]
    ts_vals = promtype_vals[promtype_vals["tss_type"] == "tissue-specific"][col]
    ub_vals = promtype_vals[promtype_vals["tss_type"] == "ubiquitous"][col]
    dy_vals = promtype_vals[promtype_vals["tss_type"] == "dynamic"][col]
    
    sns.kdeplot(data=ts_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dashed",
                label="tissue-sp.\n(n=%s)" % len(ts_vals), ax=ax)
    sns.kdeplot(data=ub_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="solid",
                label="ubiq.\n(n=%s)" % len(ub_vals), ax=ax)
    sns.kdeplot(data=dy_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dotted",
                label="dynamic\n(n=%s)" % len(dy_vals), ax=ax)
    
    ax.set_title("%s (%s total)" % (name, len(promtype_vals)))
    ax.set_xlabel(xlabel)
    ax.legend(loc="bottom right")
    ax.set_xlim(xlim)
    
    if i == 0:
        ax.set_ylabel("cumulative density")


# In[27]:


fig = plt.figure(figsize=(2, 1.7))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(max # overlapping motifs)")
ax.set_ylabel("cumulative density")
ax.set_xlim((0.2, 4.8))
ax.set_ylim((0, 1.02))
plt.legend(loc=2)
fig.savefig("max_cov.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# ### ChIP-validated motifs

# In[28]:


chip_cov["log_bp_covered"] = np.log(chip_cov["numBPcovered"]+1)
chip_cov["log_max_cov"] = np.log(chip_cov["MaxCov"]+1)

chip_cov_exp["log_bp_covered"] = np.log(chip_cov_exp["numBPcovered"]+1)
chip_cov_exp["log_max_cov"] = np.log(chip_cov_exp["MaxCov"]+1)


# In[29]:


enh_vals = chip_cov[chip_cov["PromType2"] == "Enhancer"]["log_bp_covered"]
linc_vals = chip_cov[chip_cov["PromType2"] == "intergenic"]["log_bp_covered"]
dlnc_vals = chip_cov[chip_cov["PromType2"] == "div_lnc"]["log_bp_covered"]
pc_vals = chip_cov[chip_cov["PromType2"] == "protein_coding"]["log_bp_covered"]
dpc_vals = chip_cov[chip_cov["PromType2"] == "div_pc"]["log_bp_covered"]

fig = plt.figure(figsize=(2.75, 2))
ax = sns.kdeplot(data=enh_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["Enhancer"], 
                 label="eRNAs\n(n=%s)" % len(enh_vals))
sns.kdeplot(data=linc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["intergenic"], 
            label="lincRNAs\n(n=%s)" % len(linc_vals), ax=ax)
sns.kdeplot(data=dlnc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_lnc"], 
            label="div. lncRNAs\n(n=%s)" % len(dlnc_vals), ax=ax)
sns.kdeplot(data=pc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs\n(n=%s)" % len(pc_vals), ax=ax)
sns.kdeplot(data=dpc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_pc"], 
            label="div. mRNAs\n(n=%s)" % len(dpc_vals), ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
fig.savefig("Fig_2D.pdf", bbox_inches="tight", dpi="figure")


# In[30]:


# for each group, split into tissue-sp v dynamic v ubiquitous
idxs = list(range(0, 5))
promtypes = ["Enhancer", "intergenic", "protein_coding"]
names = ["eRNAs", "lincRNAs", "mRNAs"]
df = chip_cov_exp
col = "log_bp_covered"
xlabel = "log(# of bp covered)"
xlim = (0, 4.5)

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(7.2, 2))

for i, promtype, name in zip(idxs, promtypes, names):
    print("i: %s, promtype: %s, name: %s" % (i, promtype, name))
    ax = axarr[i]
    
    promtype_vals = df[df["PromType2"] == promtype]
    ts_vals = promtype_vals[promtype_vals["tss_type"] == "tissue-specific"][col]
    ub_vals = promtype_vals[promtype_vals["tss_type"] == "ubiquitous"][col]
    dy_vals = promtype_vals[promtype_vals["tss_type"] == "dynamic"][col]
    
    sns.kdeplot(data=ts_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dashed",
                label="tissue-sp.\n(n=%s)" % len(ts_vals), ax=ax)
    sns.kdeplot(data=ub_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="solid",
                label="ubiq.\n(n=%s)" % len(ub_vals), ax=ax)
    sns.kdeplot(data=dy_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dotted",
                label="dynamic\n(n=%s)" % len(dy_vals), ax=ax)
    
    ax.set_title("%s (%s total)" % (name, len(promtype_vals)))
    ax.set_xlabel(xlabel)
    ax.legend(loc="bottom right")
    ax.set_xlim(xlim)
    
    if i == 0:
        ax.set_ylabel("cumulative density")
f.savefig("Fig_2D_biotype_split.pdf", bbox_inches="tight", dpi="figure")


# In[31]:


enh_vals = chip_cov[chip_cov["PromType2"] == "Enhancer"]["log_max_cov"]
linc_vals = chip_cov[chip_cov["PromType2"] == "intergenic"]["log_max_cov"]
dlnc_vals = chip_cov[chip_cov["PromType2"] == "div_lnc"]["log_max_cov"]
pc_vals = chip_cov[chip_cov["PromType2"] == "protein_coding"]["log_max_cov"]
dpc_vals = chip_cov[chip_cov["PromType2"] == "div_pc"]["log_max_cov"]

fig = plt.figure(figsize=(2.5, 2))
ax = sns.kdeplot(data=enh_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["Enhancer"], 
                 label="eRNAs (n=%s)" % len(enh_vals))
sns.kdeplot(data=linc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["intergenic"], 
            label="lincRNAs (n=%s)" % len(linc_vals), ax=ax)
sns.kdeplot(data=dlnc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_lnc"], 
            label="div. lncRNAs (n=%s)" % len(dlnc_vals), ax=ax)
sns.kdeplot(data=pc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs (n=%s)" % len(pc_vals), ax=ax)
sns.kdeplot(data=dpc_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE["div_pc"], 
            label="div. mRNAs (n=%s)" % len(dpc_vals), ax=ax)
ax.set_xlabel("log(max coverage)")
ax.set_ylabel("cumulative density")
fig.savefig("Fig_2E.pdf", bbox_inches="tight", dpi="figure")


# In[32]:


# for each group, split into tissue-sp v dynamic v ubiquitous
idxs = list(range(0, 5))
promtypes = ["Enhancer", "intergenic", "protein_coding"]
names = ["eRNAs", "lincRNAs", "mRNAs"]
df = chip_cov_exp
col = "log_max_cov"
xlabel = "log(max coverage)"
xlim = (-0.5, 3)

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(7.2, 2))

for i, promtype, name in zip(idxs, promtypes, names):
    print("i: %s, promtype: %s, name: %s" % (i, promtype, name))
    ax = axarr[i]
    
    promtype_vals = df[df["PromType2"] == promtype]
    ts_vals = promtype_vals[promtype_vals["tss_type"] == "tissue-specific"][col]
    ub_vals = promtype_vals[promtype_vals["tss_type"] == "ubiquitous"][col]
    dy_vals = promtype_vals[promtype_vals["tss_type"] == "dynamic"][col]
    
    sns.kdeplot(data=ts_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dashed",
                label="tissue-sp.\n(n=%s)" % len(ts_vals), ax=ax)
    sns.kdeplot(data=ub_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="solid",
                label="ubiq.\n(n=%s)" % len(ub_vals), ax=ax)
    sns.kdeplot(data=dy_vals, cumulative=True, bw=0.1, color=TSS_CLASS_PALETTE[promtype], linestyle="dotted",
                label="dynamic\n(n=%s)" % len(dy_vals), ax=ax)
    
    ax.set_title("%s (%s total)" % (name, len(promtype_vals)))
    ax.set_xlabel(xlabel)
    ax.legend(loc="bottom right")
    ax.set_xlim(xlim)
    
    if i == 0:
        ax.set_ylabel("cumulative density")
f.savefig("Fig_2E_biotype_split.pdf", bbox_inches="tight", dpi="figure")


# ## 2. cluster the motifs using MoSBAT output

# In[33]:


corr.set_index(corr["Motif"], inplace=True)
corr.drop("Motif", axis=1, inplace=True)
corr.head()


# In[34]:


row_linkage = hierarchy.linkage(distance.pdist(corr, 'correlation'), method="average")
col_linkage = hierarchy.linkage(distance.pdist(corr.T, 'correlation'), method="average")


# In[35]:


dists = plot_dendrogram(row_linkage, 0.4, "correlation")


# In[36]:


clusters = hierarchy.fcluster(row_linkage, 0.1, criterion="distance")


# In[37]:


print("n clusters: %s" % np.max(clusters))


# In[38]:


cluster_map = pd.DataFrame.from_dict(dict(zip(list(corr.index), clusters)), orient="index")
cluster_map.columns = ["cluster"]
cluster_map.head()


# ## 3. plot clustered motif heatmap

# In[39]:


colors = sns.husl_palette(np.max(clusters), s=0.75)
shuffle(colors)
lut = dict(zip(range(np.min(clusters), np.max(clusters)+1), colors))
row_colors = cluster_map["cluster"].map(lut)


# In[40]:


cmap = sns.cubehelix_palette(8, start=.5, light=1, dark=0.25, hue=0.9, rot=-0.75, as_cmap=True)


# In[41]:


cg = sns.clustermap(corr, method="average", row_linkage=row_linkage, robust=True,
                    col_linkage=col_linkage, cmap=cmap, figsize=(5, 5), row_colors=row_colors,
                    linewidths=0, rasterized=True)
cg.savefig("Fig_S7A.pdf", bbox_inches="tight", dpi="figure")


# ## 4. re-plot # bp covered and max coverage per biotype *after* clustering
# note that i sent the cluster results to marta, who re-ran her coverage scripts using them, and i re-upload them in this notebook (so in real life there is a break between the above part and the following part of this notebook)

# In[42]:


cluster_cov["log_bp_covered"] = np.log(cluster_cov["numBPcovered"]+1)
cluster_cov["log_max_cov"] = np.log(cluster_cov["MaxCov"]+1)

cluster_cov_exp["log_bp_covered"] = np.log(cluster_cov_exp["numBPcovered"]+1)
cluster_cov_exp["log_max_cov"] = np.log(cluster_cov_exp["MaxCov"]+1)


# In[43]:


enh_vals = cluster_cov[cluster_cov["PromType2"] == "Enhancer"]["log_bp_covered"]
linc_vals = cluster_cov[cluster_cov["PromType2"] == "intergenic"]["log_bp_covered"]
dlnc_vals = cluster_cov[cluster_cov["PromType2"] == "div_lnc"]["log_bp_covered"]
pc_vals = cluster_cov[cluster_cov["PromType2"] == "protein_coding"]["log_bp_covered"]
dpc_vals = cluster_cov[cluster_cov["PromType2"] == "div_pc"]["log_bp_covered"]

fig = plt.figure(figsize=(2.75, 2))
ax = sns.kdeplot(data=enh_vals, cumulative=True, color=TSS_CLASS_PALETTE["Enhancer"], 
                 label="eRNAs (n=%s)" % len(enh_vals))
sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
            label="lincRNAs (n=%s)" % len(linc_vals), ax=ax)
sns.kdeplot(data=dlnc_vals, cumulative=True, color=TSS_CLASS_PALETTE["div_lnc"], 
            label="div. lncRNAs (n=%s)" % len(dlnc_vals), ax=ax)
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs (n=%s)" % len(pc_vals), ax=ax)
sns.kdeplot(data=dpc_vals, cumulative=True, color=TSS_CLASS_PALETTE["div_pc"], 
            label="div. mRNAs (n=%s)" % len(dpc_vals), ax=ax)
ax.set_xlabel("log(# of bp covered, deduped by motif cluster)")
ax.set_ylabel("cumulative density")
plt.xlim((2.5,5))
fig.savefig("Fig_S7B.pdf", bbox_inches="tight", dpi="figure")


# In[44]:


# for each group, split into tissue-sp v dynamic v ubiquitous
idxs = list(range(0, 5))
promtypes = ["Enhancer", "intergenic", "protein_coding"]
names = ["eRNAs", "lincRNAs", "mRNAs"]
df = cluster_cov_exp
col = "log_bp_covered"
xlabel = "log(# of bp covered)"
xlim = (2, 5)

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(7.2, 2))

for i, promtype, name in zip(idxs, promtypes, names):
    print("i: %s, promtype: %s, name: %s" % (i, promtype, name))
    ax = axarr[i]
    
    promtype_vals = df[df["PromType2"] == promtype]
    ts_vals = promtype_vals[promtype_vals["tss_type"] == "tissue-specific"][col]
    ub_vals = promtype_vals[promtype_vals["tss_type"] == "ubiquitous"][col]
    dy_vals = promtype_vals[promtype_vals["tss_type"] == "dynamic"][col]
    
    sns.kdeplot(data=ts_vals, cumulative=True, color=TSS_CLASS_PALETTE[promtype], linestyle="dashed",
                label="tissue-sp.\n(n=%s)" % len(ts_vals), ax=ax)
    sns.kdeplot(data=ub_vals, cumulative=True, color=TSS_CLASS_PALETTE[promtype], linestyle="solid",
                label="ubiq.\n(n=%s)" % len(ub_vals), ax=ax)
    sns.kdeplot(data=dy_vals, cumulative=True, color=TSS_CLASS_PALETTE[promtype], linestyle="dotted",
                label="dynamic\n(n=%s)" % len(dy_vals), ax=ax)
    
    ax.set_title("%s (%s total)" % (name, len(promtype_vals)))
    ax.set_xlabel(xlabel)
    ax.legend(loc="bottom right")
    ax.set_xlim(xlim)
    
    if i == 0:
        ax.set_ylabel("cumulative density")
f.savefig("Fig_S7B_biotype_split.pdf", bbox_inches="tight", dpi="figure")


# In[45]:


enh_vals = cluster_cov[cluster_cov["PromType2"] == "Enhancer"]["log_max_cov"]
linc_vals = cluster_cov[cluster_cov["PromType2"] == "intergenic"]["log_max_cov"]
dlnc_vals = cluster_cov[cluster_cov["PromType2"] == "div_lnc"]["log_max_cov"]
pc_vals = cluster_cov[cluster_cov["PromType2"] == "protein_coding"]["log_max_cov"]
dpc_vals = cluster_cov[cluster_cov["PromType2"] == "div_pc"]["log_max_cov"]

fig = plt.figure(figsize=(2.75, 2))
ax = sns.kdeplot(data=enh_vals, cumulative=True, color=TSS_CLASS_PALETTE["Enhancer"], 
                 label="eRNAs")
sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
            label="lincRNAs", ax=ax)
sns.kdeplot(data=dlnc_vals, cumulative=True, color=TSS_CLASS_PALETTE["div_lnc"], 
            label="div. lncRNAs", ax=ax)
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
sns.kdeplot(data=dpc_vals, cumulative=True, color=TSS_CLASS_PALETTE["div_pc"], 
            label="div. mRNAs", ax=ax)
ax.set_xlabel("log(max coverage, deduped by motif cluster)")
ax.set_ylabel("cumulative density")
plt.xlim((0, 2.75))
fig.savefig("Fig_S7C.pdf", bbox_inches="tight", dpi="figure")


# In[46]:


# for each group, split into tissue-sp v dynamic v ubiquitous
idxs = list(range(0, 5))
promtypes = ["Enhancer", "intergenic", "protein_coding"]
names = ["eRNAs", "lincRNAs", "mRNAs"]
df = cluster_cov_exp
col = "log_max_cov"
xlabel = "log(max coverage)"
xlim = (0, 3)

f, axarr = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=True, figsize=(7.2, 2))

for i, promtype, name in zip(idxs, promtypes, names):
    print("i: %s, promtype: %s, name: %s" % (i, promtype, name))
    ax = axarr[i]
    
    promtype_vals = df[df["PromType2"] == promtype]
    ts_vals = promtype_vals[promtype_vals["tss_type"] == "tissue-specific"][col]
    ub_vals = promtype_vals[promtype_vals["tss_type"] == "ubiquitous"][col]
    dy_vals = promtype_vals[promtype_vals["tss_type"] == "dynamic"][col]
    
    sns.kdeplot(data=ts_vals, cumulative=True, color=TSS_CLASS_PALETTE[promtype], linestyle="dashed",
                label="tissue-sp.\n(n=%s)" % len(ts_vals), ax=ax)
    sns.kdeplot(data=ub_vals, cumulative=True, color=TSS_CLASS_PALETTE[promtype], linestyle="solid",
                label="ubiq.\n(n=%s)" % len(ub_vals), ax=ax)
    sns.kdeplot(data=dy_vals, cumulative=True, color=TSS_CLASS_PALETTE[promtype], linestyle="dotted",
                label="dynamic\n(n=%s)" % len(dy_vals), ax=ax)
    
    ax.set_title("%s (%s total)" % (name, len(promtype_vals)))
    ax.set_xlabel(xlabel)
    ax.legend(loc="bottom right")
    ax.set_xlim(xlim)
    
    if i == 0:
        ax.set_ylabel("cumulative density")
f.savefig("Fig_S7C_biotype_split.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:




