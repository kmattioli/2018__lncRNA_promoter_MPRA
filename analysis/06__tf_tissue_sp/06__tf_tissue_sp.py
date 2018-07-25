
# coding: utf-8

# # 06__tf_tissue_sp
# # calculating tissue specificity of TFs (across HepG2, HeLa, and K562)
# 
# in this notebook, i calculate the tissue specificity of TFs across the 3 cell types in our MPRAs using ENCODE RNA-seq data. then, i correlate motif coverage (# bp covered and maximum coverage) as well as average TF specificity with MPRA activities and specificities.
# 
# note: the FIMO mappings and coverage calculations were done separately (see methods)
# 
# ------
# 
# figures in this notebook:
# - **Fig 2C**: KDE plot of correlations of MPRA activity & specificity with each of the 3 metrics

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
from scipy import signal
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


index_dir = "../../data/00__index"
index_f = "%s/tss_oligo_pool.index.txt" % index_dir


# In[4]:


hepg2_activ_f = "../../data/02__activs/POOL1__pMPRA1__HepG2__activities_per_element.txt"
hela_activ_f = "../../data/02__activs/POOL1__pMPRA1__HeLa__activities_per_element.txt"
k562_activ_f = "../../data/02__activs/POOL1__pMPRA1__K562__activities_per_element.txt"


# In[5]:


fimo_f = "../../misc/03__fimo/pool1_fimo_map.txt"
fimo_chip_f = "../../misc/03__fimo/pool1_fimo_map.chip_intersected.txt"


# In[6]:


fimo_cov_f = "../../data/04__coverage/FIMO.coverage.new.txt"
fimo_chip_cov_f = "../../data/04__coverage/FIMO.ChIPIntersect.coverage.new.txt"


# In[7]:


tf_ts_f = "../../data/04__coverage/TF_tissue_specificities.from_CAGE.txt"


# In[8]:


cage_v_mpra_f = "../../data/02__activs/POOL1__pMPRA1__CAGE_vs_MPRA_activs.txt"


# ## 1. import data

# In[9]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo.columns = ["motif", "unique_id", "start", "end", "strand", "score", "pval", "qval", "seq"]
fimo.head()


# In[10]:


fimo_chip = pd.read_table(fimo_chip_f, sep="\t")
fimo_chip.head()


# In[11]:


index = pd.read_table(index_f, sep="\t")
index_elem = index[["element", "oligo_type", "unique_id", "dupe_info", "SNP", "seq_name"]]
index_elem = index_elem.drop_duplicates()


# In[12]:


hepg2_activ = pd.read_table(hepg2_activ_f, sep="\t")
hela_activ = pd.read_table(hela_activ_f, sep="\t")
k562_activ = pd.read_table(k562_activ_f, sep="\t")


# In[13]:


hepg2_reps = [x for x in hepg2_activ.columns if "rna" in x]
hela_reps = [x for x in hela_activ.columns if "rna" in x]
k562_reps = [x for x in k562_activ.columns if "rna" in x]


# In[14]:


fimo_cov = pd.read_table(fimo_cov_f, sep="\t")
fimo_cov = fimo_cov[["unique_id", "n_total_motifs", "n_unique_motifs", "max_cov", "n_bp_cov",
                     "log_n_total_motifs", "log_n_unique_motifs", "log_max_cov", "log_n_bp_cov"]]
fimo_cov.head()


# In[15]:


fimo_chip_cov = pd.read_table(fimo_chip_cov_f, sep="\t")
fimo_chip_cov = fimo_chip_cov[["unique_id", "n_total_motifs", "n_unique_motifs", "max_cov", "n_bp_cov",
                               "log_n_total_motifs", "log_n_unique_motifs", "log_max_cov", "log_n_bp_cov"]]
fimo_chip_cov.head()


# In[16]:


tf_ts = pd.read_table(tf_ts_f, sep="\t")
tf_ts.head()


# In[17]:


cage_v_mpra = pd.read_table(cage_v_mpra_f, sep="\t")
cage_v_mpra.head()


# ## 2. find avg specificity per tile

# In[18]:


fimo["motif"] = fimo["motif"].str.upper()
fimo = fimo.merge(tf_ts, left_on="motif", right_on="tf", how="left")
fimo.head()


# In[19]:


fimo_chip["motif"] = fimo_chip["motif"].str.upper()
fimo_chip = fimo_chip.merge(tf_ts, left_on="motif", right_on="tf", how="left")
fimo_chip.head()


# In[20]:


len(fimo)


# In[21]:


len(fimo_chip)


# In[22]:


fimo_nonan = fimo[~pd.isnull(fimo["tissue_sp_3"])]
len(fimo_nonan)


# In[23]:


fimo_chip_nonan = fimo_chip[~pd.isnull(fimo_chip["tissue_sp_3"])]
len(fimo_chip_nonan)


# In[24]:


fimo_deduped = fimo_nonan.drop_duplicates(subset=["motif", "unique_id"])
len(fimo_deduped)


# In[25]:


fimo_chip_deduped = fimo_chip_nonan.drop_duplicates(subset=["motif", "unique_id"])
len(fimo_chip_deduped)


# In[26]:


avg_sp_fimo = fimo_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("mean").reset_index()
avg_sp_fimo.columns = ["unique_id", "avg_tf_tissue_sp"]
avg_sp_fimo.head()


# In[27]:


avg_sp_fimo_chip = fimo_chip_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("mean").reset_index()
avg_sp_fimo_chip.columns = ["unique_id", "avg_tf_tissue_sp"]
avg_sp_fimo_chip.head()


# In[28]:


med_sp_fimo = fimo_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("median").reset_index()
med_sp_fimo.columns = ["unique_id", "med_tf_tissue_sp"]
med_sp_fimo.head()


# In[29]:


med_sp_fimo_chip = fimo_chip_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("median").reset_index()
med_sp_fimo_chip.columns = ["unique_id", "med_tf_tissue_sp"]
med_sp_fimo_chip.head()


# In[30]:


tissue_sp_fimo = avg_sp_fimo.merge(med_sp_fimo, on="unique_id")
tissue_sp_fimo["log_avg_tf_tissue_sp"] = np.log(tissue_sp_fimo["avg_tf_tissue_sp"]+1)
tissue_sp_fimo["log_med_tf_tissue_sp"] = np.log(tissue_sp_fimo["med_tf_tissue_sp"]+1)
tissue_sp_fimo.head()


# In[31]:


tissue_sp_fimo_chip = avg_sp_fimo_chip.merge(med_sp_fimo_chip, on="unique_id")
tissue_sp_fimo_chip["log_avg_tf_tissue_sp"] = np.log(tissue_sp_fimo_chip["avg_tf_tissue_sp"]+1)
tissue_sp_fimo_chip["log_med_tf_tissue_sp"] = np.log(tissue_sp_fimo_chip["med_tf_tissue_sp"]+1)
tissue_sp_fimo_chip.head()


# ## 3. find tissue specificity per tile

# In[32]:


mean_activ_fimo = cage_v_mpra.merge(tissue_sp_fimo, on="unique_id")
mean_activ_fimo.sample(5)


# In[33]:


fimo_cov.head()


# In[34]:


mean_activ_fimo = mean_activ_fimo.merge(fimo_cov, on="unique_id")
mean_activ_fimo.sample(5)


# In[35]:


mean_activ_fimo_chip = cage_v_mpra.merge(tissue_sp_fimo_chip, on="unique_id")
mean_activ_fimo_chip.sample(5)


# In[36]:


mean_activ_fimo_chip = mean_activ_fimo_chip.merge(fimo_chip_cov, on="unique_id")
mean_activ_fimo_chip.sample(5)


# ## 4. plot correlations w/ MPRA data

# ## tissue specificity

# #### fimo only

# In[37]:


#cmap = sns.light_palette("#8da0cb", as_cmap=True)
cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[38]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_activ"])]


# In[39]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_2C_3.pdf", bbox_inches="tight", dpi="figure")


# In[40]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_ts"])]


# In[41]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_2C_6.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[42]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_activ"])]


# In[43]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[44]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_ts"])]


# In[45]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# ## number of bp covered

# #### fimo only

# In[46]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_activ"])]


# In[47]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_2C_1.pdf", bbox_inches="tight", dpi="figure")


# In[48]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_ts"])]


# In[49]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_2C_4.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[50]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_activ"])]


# In[51]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[52]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_ts"])]


# In[53]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# ## max overlapping coverage

# #### fimo only

# In[54]:


cmap = sns.light_palette("firebrick", as_cmap=True)


# In[55]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_activ"])]


# In[56]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_2C_2.pdf", bbox_inches="tight", dpi="figure")


# In[57]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_ts"])]


# In[58]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_2C_5.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[59]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_activ"])]


# In[60]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[61]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_ts"])]


# In[62]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# ## 5. plot correlations w/ CAGE data

# ## TF tissue specificity

# #### fimo only

# In[63]:


#cmap = sns.light_palette("#8da0cb", as_cmap=True)
cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[64]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["cage_activ"])]


# In[65]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[66]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[67]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# #### fimo intersected w/ chip

# In[68]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_activ"])]


# In[69]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[70]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_ts"])]


# In[71]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# ## # bp covered

# #### fimo only

# In[72]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_activ"])]


# In[73]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[74]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[75]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# #### fimo intersected w/ chip

# In[76]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_activ"])]


# In[77]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[78]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_ts"])]


# In[79]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# ## max overlapping motifs

# #### fimo only

# In[80]:


cmap = sns.light_palette("firebrick", as_cmap=True)


# In[81]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_activ"])]


# In[82]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[83]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[84]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# #### fimo intersected w/ chip

# In[85]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_activ"])]


# In[86]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[87]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_ts"])]


# In[88]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# ## 6. write files

# In[89]:


mean_activ_fimo.drop(["med_tf_tissue_sp", "log_med_tf_tissue_sp"], axis=1, inplace=True)


# In[90]:


mean_activ_fimo_chip.drop(["med_tf_tissue_sp", "log_med_tf_tissue_sp"], axis=1, inplace=True)


# In[91]:


out_dir = "../../data/04__coverage"
get_ipython().system('mkdir -p $out_dir')
mean_activ_fimo.to_csv("%s/FIMO.coverage.new.txt" % out_dir, sep="\t", index=False)
mean_activ_fimo_chip.to_csv("%s/FIMO.ChIPIntersect.coverage.new.txt" % out_dir, sep="\t", index=False)


# In[ ]:




