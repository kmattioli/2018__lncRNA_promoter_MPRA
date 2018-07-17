
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
# - **Fig 3C**: KDE plot of correlations of MPRA activity & specificity with each of the 3 metrics

# In[ ]:


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


# In[ ]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## variables

# In[ ]:


index_dir = "../../data/00__index"
index_f = "%s/tss_oligo_pool.index.txt" % index_dir


# In[ ]:


hepg2_activ_f = "../../data/02__activs/POOL1__pMPRA1__HepG2__activities_per_element.txt"
hela_activ_f = "../../data/02__activs/POOL1__pMPRA1__HeLa__activities_per_element.txt"
k562_activ_f = "../../data/02__activs/POOL1__pMPRA1__K562__activities_per_element.txt"


# In[ ]:


fimo_f = "../../misc/05__fimo/pool1_fimo_map.txt"
fimo_chip_f = "../../misc/05__fimo/pool1_fimo_map.chip_intersected.txt"


# In[ ]:


fimo_cov_f = "../../data/04__coverage/FIMO.coverage.new.txt"
fimo_chip_cov_f = "../../data/04__coverage/FIMO.ChIPIntersect.coverage.new.txt"


# In[ ]:


tf_ts_f = "../../data/04__coverage/TF_tissue_specificities.from_CAGE.txt"


# In[ ]:


cage_v_mpra_f = "../../data/02__activs/POOL1__pMPRA1__CAGE_vs_MPRA_activs.txt"


# ## 1. import data

# In[ ]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo.columns = ["motif", "unique_id", "start", "end", "strand", "score", "pval", "qval", "seq"]
fimo.head()


# In[ ]:


fimo_chip = pd.read_table(fimo_chip_f, sep="\t")
fimo_chip.head()


# In[ ]:


index = pd.read_table(index_f, sep="\t")
index_elem = index[["element", "oligo_type", "unique_id", "dupe_info", "SNP", "seq_name"]]
index_elem = index_elem.drop_duplicates()


# In[ ]:


hepg2_activ = pd.read_table(hepg2_activ_f, sep="\t")
hela_activ = pd.read_table(hela_activ_f, sep="\t")
k562_activ = pd.read_table(k562_activ_f, sep="\t")


# In[ ]:


hepg2_reps = [x for x in hepg2_activ.columns if "rna" in x]
hela_reps = [x for x in hela_activ.columns if "rna" in x]
k562_reps = [x for x in k562_activ.columns if "rna" in x]


# In[ ]:


fimo_cov = pd.read_table(fimo_cov_f, sep="\t")
fimo_cov.head()


# In[ ]:


fimo_chip_cov = pd.read_table(fimo_chip_cov_f, sep="\t")
fimo_chip_cov.head()


# In[ ]:


tf_ts = pd.read_table(tf_ts_f, sep="\t")
tf_ts.head()


# In[ ]:


cage_v_mpra = pd.read_table(cage_v_mpra_f, sep="\t")
cage_v_mpra.head()


# ## 2. find avg specificity per tile

# In[ ]:


fimo["motif"] = fimo["motif"].str.upper()
fimo = fimo.merge(tf_ts, left_on="motif", right_on="tf", how="left")
fimo.head()


# In[ ]:


fimo_chip["motif"] = fimo_chip["motif"].str.upper()
fimo_chip = fimo_chip.merge(tf_ts, left_on="motif", right_on="tf", how="left")
fimo_chip.head()


# In[ ]:


len(fimo)


# In[ ]:


len(fimo_chip)


# In[ ]:


fimo_nonan = fimo[~pd.isnull(fimo["tissue_sp_3"])]
len(fimo_nonan)


# In[ ]:


fimo_chip_nonan = fimo_chip[~pd.isnull(fimo_chip["tissue_sp_3"])]
len(fimo_chip_nonan)


# In[ ]:


fimo_deduped = fimo_nonan.drop_duplicates(subset=["motif", "unique_id"])
len(fimo_deduped)


# In[ ]:


fimo_chip_deduped = fimo_chip_nonan.drop_duplicates(subset=["motif", "unique_id"])
len(fimo_chip_deduped)


# In[ ]:


avg_sp_fimo = fimo_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("mean").reset_index()
avg_sp_fimo.columns = ["unique_id", "avg_tf_tissue_sp"]
avg_sp_fimo.head()


# In[ ]:


avg_sp_fimo_chip = fimo_chip_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("mean").reset_index()
avg_sp_fimo_chip.columns = ["unique_id", "avg_tf_tissue_sp"]
avg_sp_fimo_chip.head()


# In[ ]:


med_sp_fimo = fimo_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("median").reset_index()
med_sp_fimo.columns = ["unique_id", "med_tf_tissue_sp"]
med_sp_fimo.head()


# In[ ]:


med_sp_fimo_chip = fimo_chip_deduped.groupby(["unique_id"])["tissue_sp_3"].agg("median").reset_index()
med_sp_fimo_chip.columns = ["unique_id", "med_tf_tissue_sp"]
med_sp_fimo_chip.head()


# In[ ]:


tissue_sp_fimo = avg_sp_fimo.merge(med_sp_fimo, on="unique_id")
tissue_sp_fimo["log_avg_tf_tissue_sp"] = np.log(tissue_sp_fimo["avg_tf_tissue_sp"]+1)
tissue_sp_fimo["log_med_tf_tissue_sp"] = np.log(tissue_sp_fimo["med_tf_tissue_sp"]+1)
tissue_sp_fimo.head()


# In[ ]:


tissue_sp_fimo_chip = avg_sp_fimo_chip.merge(med_sp_fimo_chip, on="unique_id")
tissue_sp_fimo_chip["log_avg_tf_tissue_sp"] = np.log(tissue_sp_fimo_chip["avg_tf_tissue_sp"]+1)
tissue_sp_fimo_chip["log_med_tf_tissue_sp"] = np.log(tissue_sp_fimo_chip["med_tf_tissue_sp"]+1)
tissue_sp_fimo_chip.head()


# ## 3. find tissue specificity per tile

# In[ ]:


mean_activ_fimo = cage_v_mpra.merge(tissue_sp_fimo, on="unique_id")
mean_activ_fimo.sample(5)


# In[ ]:


mean_activ_fimo = mean_activ_fimo.merge(fimo_cov, left_on="unique_id", right_on="index")
mean_activ_fimo.sample(5)


# In[ ]:


mean_activ_fimo_chip = cage_v_mpra.merge(tissue_sp_fimo_chip, on="unique_id")
mean_activ_fimo_chip.sample(5)


# In[ ]:


mean_activ_fimo_chip = mean_activ_fimo_chip.merge(fimo_chip_cov, left_on="unique_id", right_on="index")
mean_activ_fimo_chip.sample(5)


# ## 4. plot correlations w/ MPRA data

# ## tissue specificity

# #### fimo only

# In[ ]:


#cmap = sns.light_palette("#8da0cb", as_cmap=True)
cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_3.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_3.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# ## number of bp covered

# #### fimo only

# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_1.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_1.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# ## mean overlapping coverage

# #### fimo only

# In[ ]:


cmap = sns.light_palette("firebrick", as_cmap=True)


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_mean_cov"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_2.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_2.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# ## 5. plot correlations w/ CAGE data

# ## TF tissue specificity

# #### fimo only

# In[ ]:


#cmap = sns.light_palette("#8da0cb", as_cmap=True)
cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["cage_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_avg_tf_tissue_sp"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_avg_tf_tissue_sp"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# ## # bp covered

# #### fimo only

# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_n_bp_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_n_bp_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["cage_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# ## mean overlapping motifs

# #### fimo only

# In[ ]:


cmap = sns.light_palette("firebrick", as_cmap=True)


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["log_max_cov"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# #### fimo intersected w/ chip

# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("CAGE expression")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = mean_activ_fimo_chip[~pd.isnull(mean_activ_fimo_chip["log_max_cov"]) &
                              ~pd.isnull(mean_activ_fimo_chip["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# ### check correlation b/w CAGE and MPRA ts

# In[ ]:


no_nan = mean_activ_fimo[~pd.isnull(mean_activ_fimo["mpra_ts"]) &
                         ~pd.isnull(mean_activ_fimo["cage_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["mpra_ts"], no_nan["cage_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in CAGE")
ax.set_xlabel("tissue specificity in MPRA")

r, p = stats.spearmanr(no_nan["mpra_ts"], no_nan["cage_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# ## 6. control for # of motifs

# In[ ]:


sns.distplot(mean_activ_fimo[["n_total_motifs"]], kde=False, bins=50)


# In[ ]:


len(mean_activ_fimo[(mean_activ_fimo["n_total_motifs"] >= 20) & (mean_activ_fimo["n_total_motifs"] <= 25)])


# In[ ]:


sampled = mean_activ_fimo[(mean_activ_fimo["n_total_motifs"] >= 20) & (mean_activ_fimo["n_total_motifs"] <= 25)]


# ## tissue specificity

# #### fimo only

# In[ ]:


#cmap = sns.light_palette("#8da0cb", as_cmap=True)
cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[ ]:


no_nan = sampled[~pd.isnull(sampled["log_avg_tf_tissue_sp"]) &
                 ~pd.isnull(sampled["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_3.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = sampled[~pd.isnull(sampled["log_avg_tf_tissue_sp"]) &
                 ~pd.isnull(sampled["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(no_nan["log_avg_tf_tissue_sp"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# ## number of bp covered

# #### fimo only

# In[ ]:


no_nan = sampled[~pd.isnull(sampled["log_n_bp_cov"]) &
                 ~pd.isnull(sampled["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_activ"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_1.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = sampled[~pd.isnull(sampled["log_n_bp_cov"]) &
                 ~pd.isnull(sampled["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_n_bp_cov"], no_nan["mpra_ts"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(no_nan["log_n_bp_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# ## mean overlapping coverage

# #### fimo only

# In[ ]:


cmap = sns.light_palette("firebrick", as_cmap=True)


# In[ ]:


no_nan = sampled[~pd.isnull(sampled["log_max_cov"]) &
                 ~pd.isnull(sampled["mpra_activ"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_activ"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_activ"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_2.pdf", bbox_inches="tight", dpi="figure")


# In[ ]:


no_nan = sampled[~pd.isnull(sampled["log_max_cov"]) &
                 ~pd.isnull(sampled["mpra_ts"])]


# In[ ]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(no_nan["log_max_cov"], no_nan["mpra_ts"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(max overlapping motifs)")

r, p = stats.spearmanr(no_nan["log_max_cov"], no_nan["mpra_ts"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
#fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


# ## 7. write files

# In[44]:


final = mean_activ[["unique_id", "tile_mean_expr", "tile_tissue_sp", "avg_tf_tissue_sp", "log_avg_tf_tissue_sp",
                    "numMotifs", "log_num_motifs", "numBPcovered", "log_bp_cov", "maxCov", "log_max_cov"]]
final.head()


# In[45]:


final.columns = ["unique_id", "MPRA_mean_activ", "MPRA_tissue_sp", "avg_tf_tissue_sp", "log_avg_tf_tissue_sp",
                 "num_motifs", "log_num_motifs", "num_bp_covered", "log_num_bp_covered", "max_coverage",
                 "log_max_coverage"]
final.head()


# In[46]:


out_dir = "../../data/04__coverage"
get_ipython().system('mkdir -p $out_dir')
final.to_csv("%s/motif_coverage.txt" % out_dir, sep="\t", index=False)
tf_expr.to_csv("%s/tf_tissue_sp.txt" % out_dir, sep="\t", index=False)


# In[ ]:




