
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


expr_f = "../../misc/03__rna_seq_expr/Expression.all.cells.txt"
tf_map_f = "../../misc/04__jaspar_id_map/2018_03_09_gencode_jaspar_curated.txt"
fimo_f = "../../misc/05__fimo/pool1_fimo_map.txt"
cov_map_f = "../../misc/05__fimo/seqID_nFimoMotifs_Coverage.txt"


# In[4]:


index_dir = "../../data/00__index"
index_f = "%s/tss_oligo_pool.index.txt" % index_dir


# In[5]:


hepg2_activ_f = "../../data/02__activs/POOL1__pMPRA1__HepG2__activities_per_element.txt"
hela_activ_f = "../../data/02__activs/POOL1__pMPRA1__HeLa__activities_per_element.txt"
k562_activ_f = "../../data/02__activs/POOL1__pMPRA1__K562__activities_per_element.txt"


# ## 1. import data

# In[6]:


tf_map = pd.read_table(tf_map_f)
tf_map.head()


# In[7]:


expr = pd.read_table(expr_f)
expr.head()


# In[8]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo.columns = ["motif", "unique_id", "start", "end", "strand", "score", "pval", "qval", "seq"]
fimo.head()


# In[9]:


cov_map = pd.read_table(cov_map_f)
cov_map.head()


# In[10]:


index = pd.read_table(index_f, sep="\t")
index_elem = index[["element", "oligo_type", "unique_id", "dupe_info", "SNP", "seq_name"]]
index_elem = index_elem.drop_duplicates()


# In[11]:


hepg2_activ = pd.read_table(hepg2_activ_f, sep="\t")
hela_activ = pd.read_table(hela_activ_f, sep="\t")
k562_activ = pd.read_table(k562_activ_f, sep="\t")


# In[12]:


hepg2_reps = [x for x in hepg2_activ.columns if "rna" in x]
hela_reps = [x for x in hela_activ.columns if "rna" in x]
k562_reps = [x for x in k562_activ.columns if "rna" in x]


# ## 2. find expr of TFs in HeLa, HepG2, K562

# In[13]:


expr["ensembl_id"] = expr["gene_id"].str.split(".", expand=True)[0]
expr.head()


# In[14]:


tf_expr = tf_map.merge(expr, left_on="Gene ID", right_on="ensembl_id", how="left")
tf_expr.head()


# In[15]:


tf_expr = tf_expr[["motif_id", "motif_name", "ensembl_id", "HeLa-S3", "HepG2", "K562"]]


# ## 3. calculate tissue-sp of TFs

# In[16]:


specificities = calculate_tissue_specificity(tf_expr[["HepG2", "HeLa-S3", "K562"]])
tf_expr["tissue_sp"] = specificities
tf_expr.head()


# ## 4. find avg specificity per tile

# In[17]:


fimo = fimo.merge(tf_expr, left_on="motif", right_on="motif_name", how="left")
fimo.head()


# In[18]:


len(fimo)


# In[19]:


fimo_nonan = fimo[~pd.isnull(fimo["tissue_sp"])]
len(fimo_nonan)


# In[20]:


fimo_deduped = fimo_nonan.drop_duplicates(subset=["motif", "unique_id"])
len(fimo_deduped)


# In[21]:


avg_sp = fimo_deduped.groupby(["unique_id"])["tissue_sp"].agg("mean").reset_index()
avg_sp.columns = ["unique_id", "avg_tf_tissue_sp"]
avg_sp.head()


# In[22]:


med_sp = fimo_deduped.groupby(["unique_id"])["tissue_sp"].agg("median").reset_index()
med_sp.columns = ["unique_id", "med_tf_tissue_sp"]
med_sp.head()


# In[23]:


tissue_sp = avg_sp.merge(med_sp, on="unique_id")
tissue_sp["log_avg_tf_tissue_sp"] = np.log(tissue_sp["avg_tf_tissue_sp"]+1)
tissue_sp["log_med_tf_tissue_sp"] = np.log(tissue_sp["med_tf_tissue_sp"]+1)
tissue_sp.head()


# ## 5. find tissue specificity per tile

# In[24]:


hepg2_activ["overall_mean"] = hepg2_activ.mean(axis=1)
hela_activ["overall_mean"] = hela_activ.mean(axis=1)
k562_activ["overall_mean"] = k562_activ.mean(axis=1)


# In[25]:


mean_activ = hepg2_activ[["unique_id", "overall_mean"]].merge(hela_activ[["unique_id", "overall_mean"]], on="unique_id").merge(k562_activ[["unique_id", "overall_mean"]], on="unique_id")
mean_activ.columns = ["unique_id", "hepg2_mean", "hela_mean", "k562_mean"]
mean_activ.head()


# In[26]:


mean_activ = mean_activ[~(mean_activ["unique_id"].str.contains("SNP_INDIV")) &
                        ~(mean_activ["unique_id"].str.contains("SNP_PLUS_HAPLO")) & 
                        ~(mean_activ["unique_id"].str.contains("SCRAMBLED")) &
                        ~(mean_activ["unique_id"].str.contains("FLIPPED")) &
                        ~(mean_activ["unique_id"].str.contains("RANDOM"))]
mean_activ.sample(5)


# In[27]:


mean_activ["tile_mean_expr"] = mean_activ[["hepg2_mean", "hela_mean", "k562_mean"]].mean(axis=1)
mean_activ.head()


# In[28]:


mean_activ = mean_activ.merge(tissue_sp, on="unique_id", how="left")
mean_activ.head()


# In[29]:


# first scale ranges to be positive
mean_activ["hepg2_scaled"] = scale_range(mean_activ["hepg2_mean"], 0, 100)
mean_activ["hela_scaled"] = scale_range(mean_activ["hela_mean"], 0, 100)
mean_activ["k562_scaled"] = scale_range(mean_activ["k562_mean"], 0, 100)


# In[30]:


specificities = calculate_tissue_specificity(mean_activ[["hepg2_scaled", "hela_scaled", "k562_scaled"]])
mean_activ["tile_tissue_sp"] = specificities
mean_activ.head()


# In[31]:


mean_activ = mean_activ.merge(cov_map, left_on="unique_id", right_on="seqID", how="left")
mean_activ.head()


# In[32]:


mean_activ["log_max_cov"] = np.log(mean_activ["maxCov"]+1)
mean_activ["log_num_motifs"] = np.log(mean_activ["numMotifs"]+1)
mean_activ["log_bp_cov"] = np.log(mean_activ["numBPcovered"]+1)


# ## 6. plot correlations

# In[33]:


for_joint = mean_activ[["unique_id", "tile_mean_expr", "tile_tissue_sp", "avg_tf_tissue_sp", "med_tf_tissue_sp", 
                        "log_avg_tf_tissue_sp", "log_med_tf_tissue_sp"]]
for_joint.set_index("unique_id", inplace=True)
for_joint.head()


# ## tissue specificity

# In[34]:


for_joint.dropna(inplace=True)


# In[35]:


#cmap = sns.light_palette("#8da0cb", as_cmap=True)
cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[36]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(for_joint["log_avg_tf_tissue_sp"], for_joint["tile_mean_expr"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(for_joint["log_avg_tf_tissue_sp"], for_joint["tile_mean_expr"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_3.pdf", bbox_inches="tight", dpi="figure")


# In[37]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(for_joint["log_avg_tf_tissue_sp"], for_joint["tile_tissue_sp"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(mean TF tissue specificity)")

r, p = stats.spearmanr(for_joint["log_avg_tf_tissue_sp"], for_joint["tile_tissue_sp"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_6.pdf", bbox_inches="tight", dpi="figure")


# ## number of bp covered

# In[38]:


mean_activ.dropna(inplace=True)


# In[39]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(mean_activ["log_bp_cov"], mean_activ["tile_mean_expr"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(mean_activ["log_bp_cov"], mean_activ["tile_mean_expr"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_1.pdf", bbox_inches="tight", dpi="figure")


# In[40]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(mean_activ["log_bp_cov"], mean_activ["tile_tissue_sp"], cmap="Blues", 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(number of bp covered by motif)")

r, p = stats.spearmanr(mean_activ["log_bp_cov"], mean_activ["tile_tissue_sp"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_4.pdf", bbox_inches="tight", dpi="figure")


# ## max coverage

# In[41]:


cmap = sns.light_palette("firebrick", as_cmap=True)


# In[42]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(mean_activ["log_max_cov"], mean_activ["tile_mean_expr"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("mean MPRA activity")
ax.set_xlabel("log(maximum motif coverage)")

r, p = stats.spearmanr(mean_activ["log_max_cov"], mean_activ["tile_mean_expr"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_2.pdf", bbox_inches="tight", dpi="figure")


# In[43]:


fig = plt.figure(figsize=(1.2, 1.2))
ax = sns.kdeplot(mean_activ["log_max_cov"], mean_activ["tile_tissue_sp"], cmap=cmap, 
                 shade=True, shade_lowest=False)
ax.set_ylabel("tissue specificity in MPRA")
ax.set_xlabel("log(maximum motif coverage)")

r, p = stats.spearmanr(mean_activ["log_max_cov"], mean_activ["tile_tissue_sp"])
print("r: %s, spearman p: %s" % (r, p))
ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)
fig.savefig("Fig_3C_5.pdf", bbox_inches="tight", dpi="figure")


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

