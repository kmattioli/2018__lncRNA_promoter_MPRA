
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


# In[10]:


motif_cov_f = "../../misc/03__fimo/All.TSS.114bp.Motifs.txt"
motif_max_f = "../../misc/03__fimo/All.TSS.114bp.maxonly.txt"
chip_cov_f = "../../misc/03__fimo/All.TSS.114bp.Motifs.Intersect.Chip.ALL.txt"
cluster_cov_f = "../../misc/03__fimo/All.TSS.114bp.Cluster.ALL.txt"


# ## 1. import data

# In[6]:


corr = pd.read_table(mosbat_file, sep="\t")


# In[14]:


motif_cov = pd.read_table(motif_cov_f, sep="\t")
motif_max = pd.read_table(motif_max_f, sep="\t", header=None)
motif_max.columns = ["seqID", "MaxCov"]
motif_cov = motif_cov.merge(motif_max, on="seqID")
motif_cov.head()


# In[9]:


chip_cov = pd.read_table(chip_cov_f, sep="\t")
cluster_cov = pd.read_table(cluster_cov_f, sep="\t")
chip_cov.head()


# In[17]:


motif_cov = motif_cov.merge(chip_cov[["seqID", "PromType2"]], on="seqID")
motif_cov.head()


# ## 2. plot # bp covered & max cov across biotypes

# ### all motifs

# In[44]:


motif_cov["log_bp_covered"] = np.log(motif_cov["numBPcovered"]+1)
motif_cov["log_max_cov"] = np.log(motif_cov["MaxCov"]+1)


# In[46]:


enh_vals = motif_cov[motif_cov["PromType2"] == "Enhancer"]["log_bp_covered"]
linc_vals = motif_cov[motif_cov["PromType2"] == "intergenic"]["log_bp_covered"]
dlnc_vals = motif_cov[motif_cov["PromType2"] == "div_lnc"]["log_bp_covered"]
pc_vals = motif_cov[motif_cov["PromType2"] == "protein_coding"]["log_bp_covered"]
dpc_vals = motif_cov[motif_cov["PromType2"] == "div_pc"]["log_bp_covered"]

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
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")


# In[47]:


fig = plt.figure(figsize=(2.4, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=False, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs", shade=True)
sns.kdeplot(data=pc_vals, cumulative=False, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax, shade=True)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("density")
ax.set_xlim((2, 5.5))
ax.set_ylim((0, 2))
fig.savefig("num_bp_cov.kde.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[48]:


fig = plt.figure(figsize=(1.5, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
ax.set_xlim((2, 5))
ax.set_ylim((0, 1.02))
fig.savefig("num_bp_cov.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[49]:


enh_vals = motif_cov[motif_cov["PromType2"] == "Enhancer"]["log_max_cov"]
linc_vals = motif_cov[motif_cov["PromType2"] == "intergenic"]["log_max_cov"]
dlnc_vals = motif_cov[motif_cov["PromType2"] == "div_lnc"]["log_max_cov"]
pc_vals = motif_cov[motif_cov["PromType2"] == "protein_coding"]["log_max_cov"]
dpc_vals = motif_cov[motif_cov["PromType2"] == "div_pc"]["log_max_cov"]

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
ax.set_xlabel("log(max coverage)")
ax.set_ylabel("cumulative density")


# In[50]:


fig = plt.figure(figsize=(2.4, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=False, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs", shade=True)
sns.kdeplot(data=pc_vals, cumulative=False, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax, shade=True)
ax.set_xlabel("log(max # overlapping motifs)")
ax.set_ylabel("density")
ax.set_ylim((0, 0.7))
fig.savefig("max_cov.kde.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[53]:


fig = plt.figure(figsize=(1.5, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
ax.set_xlim((0.2, 4.8))
ax.set_ylim((0, 1.02))
plt.legend(loc=2)
fig.savefig("max_cov.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# ### ChIP-validated motifs

# In[54]:


chip_cov["log_bp_covered"] = np.log(chip_cov["numBPcovered"]+1)
chip_cov["log_max_cov"] = np.log(chip_cov["MaxCov"]+1)


# In[55]:


enh_vals = chip_cov[chip_cov["PromType2"] == "Enhancer"]["log_bp_covered"]
linc_vals = chip_cov[chip_cov["PromType2"] == "intergenic"]["log_bp_covered"]
dlnc_vals = chip_cov[chip_cov["PromType2"] == "div_lnc"]["log_bp_covered"]
pc_vals = chip_cov[chip_cov["PromType2"] == "protein_coding"]["log_bp_covered"]
dpc_vals = chip_cov[chip_cov["PromType2"] == "div_pc"]["log_bp_covered"]

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
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
fig.savefig("Fig_2D.pdf", bbox_inches="tight", dpi="figure")


# In[57]:


fig = plt.figure(figsize=(2.4, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=False, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs", shade=True)
sns.kdeplot(data=pc_vals, cumulative=False, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax, shade=True)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("density")
#ax.set_xlim((2, 5.5))
#ax.set_ylim((0, 2))
fig.savefig("num_bp_cov_chip.kde.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[58]:


fig = plt.figure(figsize=(1.5, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
#ax.set_xlim((2, 5))
ax.set_ylim((0, 1.02))
fig.savefig("num_bp_cov_chip.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[60]:


enh_vals = chip_cov[chip_cov["PromType2"] == "Enhancer"]["log_max_cov"]
linc_vals = chip_cov[chip_cov["PromType2"] == "intergenic"]["log_max_cov"]
dlnc_vals = chip_cov[chip_cov["PromType2"] == "div_lnc"]["log_max_cov"]
pc_vals = chip_cov[chip_cov["PromType2"] == "protein_coding"]["log_max_cov"]
dpc_vals = chip_cov[chip_cov["PromType2"] == "div_pc"]["log_max_cov"]

fig = plt.figure(figsize=(2.5, 2))
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
ax.set_xlabel("log(max coverage)")
ax.set_ylabel("cumulative density")
fig.savefig("Fig_2E.pdf", bbox_inches="tight", dpi="figure")


# In[61]:


fig = plt.figure(figsize=(2.4, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=False, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs", shade=True)
sns.kdeplot(data=pc_vals, cumulative=False, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax, shade=True)
ax.set_xlabel("log(max # overlapping motifs)")
ax.set_ylabel("density")
#ax.set_ylim((0, 0.7))
fig.savefig("max_cov_chip.kde.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[63]:


fig = plt.figure(figsize=(1.5, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
#ax.set_xlim((0.2, 4.8))
#ax.set_ylim((0, 1.02))
plt.legend(loc=4)
fig.savefig("max_cov_chip.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# ## 2. cluster the motifs using MoSBAT output

# In[64]:


corr.set_index(corr["Motif"], inplace=True)
corr.drop("Motif", axis=1, inplace=True)
corr.head()


# In[65]:


row_linkage = hierarchy.linkage(distance.pdist(corr, 'correlation'), method="average")
col_linkage = hierarchy.linkage(distance.pdist(corr.T, 'correlation'), method="average")


# In[66]:


dists = plot_dendrogram(row_linkage, 0.4, "correlation")


# In[67]:


clusters = hierarchy.fcluster(row_linkage, 0.1, criterion="distance")


# In[68]:


print("n clusters: %s" % np.max(clusters))


# In[69]:


cluster_map = pd.DataFrame.from_dict(dict(zip(list(corr.index), clusters)), orient="index")
cluster_map.columns = ["cluster"]
cluster_map.head()


# ## 3. plot clustered motif heatmap

# In[70]:


colors = sns.husl_palette(np.max(clusters), s=0.75)
shuffle(colors)
lut = dict(zip(range(np.min(clusters), np.max(clusters)+1), colors))
row_colors = cluster_map["cluster"].map(lut)


# In[71]:


cmap = sns.cubehelix_palette(8, start=.5, light=1, dark=0.25, hue=0.9, rot=-0.75, as_cmap=True)


# In[72]:


cg = sns.clustermap(corr, method="average", row_linkage=row_linkage, robust=True,
                    col_linkage=col_linkage, cmap=cmap, figsize=(5, 5), row_colors=row_colors,
                    linewidths=0, rasterized=True)
cg.savefig("Fig_S7A.pdf", bbox_inches="tight", dpi="figure")


# ## 4. re-plot # bp covered and max coverage per biotype *after* clustering
# note that i sent the cluster results to marta, who re-ran her coverage scripts using them, and i re-upload them in this notebook (so in real life there is a break between the above part and the following part of this notebook)

# In[73]:


cluster_cov["log_bp_covered"] = np.log(cluster_cov["numBPcovered"]+1)
cluster_cov["log_max_cov"] = np.log(cluster_cov["MaxCov"]+1)


# In[74]:


enh_vals = cluster_cov[cluster_cov["PromType2"] == "Enhancer"]["log_bp_covered"]
linc_vals = cluster_cov[cluster_cov["PromType2"] == "intergenic"]["log_bp_covered"]
dlnc_vals = cluster_cov[cluster_cov["PromType2"] == "div_lnc"]["log_bp_covered"]
pc_vals = cluster_cov[cluster_cov["PromType2"] == "protein_coding"]["log_bp_covered"]
dpc_vals = cluster_cov[cluster_cov["PromType2"] == "div_pc"]["log_bp_covered"]

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
ax.set_xlabel("log(# of bp covered, deduped by motif cluster)")
ax.set_ylabel("cumulative density")
plt.xlim((2.5,5))
fig.savefig("Fig_S7B.pdf", bbox_inches="tight", dpi="figure")


# In[79]:


fig = plt.figure(figsize=(2.4, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=False, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs", shade=True)
sns.kdeplot(data=pc_vals, cumulative=False, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax, shade=True)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("density")
ax.set_xlim((2, 5.1))
ax.set_ylim((0, 1.5))
fig.savefig("num_bp_cov_cluster.kde.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[82]:


fig = plt.figure(figsize=(1.5, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
ax.set_xlim((1, 5))
ax.set_ylim((0, 1.02))
fig.savefig("num_bp_cov_cluster.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[83]:


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


# In[84]:


fig = plt.figure(figsize=(2.4, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=False, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs", shade=True)
sns.kdeplot(data=pc_vals, cumulative=False, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax, shade=True)
ax.set_xlabel("log(max # overlapping motifs)")
ax.set_ylabel("density")
#ax.set_ylim((0, 0.7))
fig.savefig("max_cov_cluster.kde.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[86]:


fig = plt.figure(figsize=(1.5, 1.2))
ax = sns.kdeplot(data=linc_vals, cumulative=True, color=TSS_CLASS_PALETTE["intergenic"], 
                 label="lincRNAs")
sns.kdeplot(data=pc_vals, cumulative=True, color=TSS_CLASS_PALETTE["protein_coding"], 
            label="mRNAs", ax=ax)
ax.set_xlabel("log(# of bp covered)")
ax.set_ylabel("cumulative density")
ax.set_xlim((0, 2.7))
#ax.set_ylim((0, 1.02))
plt.legend(loc=2)
fig.savefig("max_cov_cluster.cdf.for_talk.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




