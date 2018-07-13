
# coding: utf-8

# # 13__motif_chip_tf_ts_redo
# # testing whether gene tissue-sp correlates with chip/motif tissue-sp
# 
# after talking to lucas: find tissue-sp for each TF, find all genes that have that motif/chip peak, calc. average tissue-sp. of those genes, correlate that with the tissue-sp. (and subset by class, if needed)
# 
# chip and fimo files are from PJ
# 
# took fimo mappings from marta's directory

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


# In[2]:


sns.set(**PAPER_PRESET)
fontsize = PAPER_FONTSIZE


# ## functions

# In[3]:


def get_tss_id(row):
    if "Enhancer" in row.tss_name:
        return row.tss_name.split("__")[1]
    else:
        return row.tss_name.split("__")[2]


# ## variables
# ### tissue specificities

# In[4]:


tss_ts_f = "TSS.CAGE_grouped_exp.tissue_sp.txt"
enh_ts_f = "Enh.CAGE_grouped_exp.tissue_sp.txt"


# In[5]:


tf_ts_f = "TF_tissue_specificities.from_CAGE.txt"


# ### for all promoters (3kb)

# In[6]:


chip_f = "chip_all.txt"


# In[7]:


fimo_f = "fimo_all_biotypes.txt"


# In[8]:


annot_f = "../../misc/00__tss_properties/TSS_FantomCat_all.TSSperENSG.txt"


# ### for all promoters (114bp)

# In[9]:


chip_114_f = "chip_all_114.txt"


# In[10]:


fimo_114_f = "../../misc/05__fimo/TFmotifs__intersect_114bpTSS.uniq.txt"


# ### pool1 tss

# In[11]:


pool1_annot_f = "../../misc/00__tss_properties/TABLE_ALL_TSS_and_flipped.properties.PromType.txt"


# ## 1. import data

# In[12]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo = fimo[fimo["shuffled"] != "shuffled"]
fimo.head()


# In[13]:


chip = pd.read_table(chip_f, sep="\t")
chip.head()


# In[14]:


annot = pd.read_table(annot_f, sep="\t")
promtype2 = annot[["gene_id", "PromType2"]].drop_duplicates()
promtype2.head()


# In[15]:


tf_ts = pd.read_table(tf_ts_f, sep="\t")
tf_ts.head()


# In[20]:


tss_ts = pd.read_table(tss_ts_f, sep="\t")
tss_ts.head()


# In[17]:


enh_ts = pd.read_table(enh_ts_f, sep="\t")
enh_ts.head()


# In[21]:


tss_ts.drop("short_description", axis=1, inplace=True)
tss_ts_cols = ["Id"]
sample_cols = [x for x in tss_ts.columns if "Group_" in x or "tissue_sp_" in x]
tss_ts_cols.extend(sample_cols)
tss_ts.columns = tss_ts_cols
tss_ts.head()


# In[22]:


all_ts = tss_ts.append(enh_ts)
all_ts.sample(5)


# In[23]:


chip_114 = pd.read_table(chip_114_f, sep="\t", header=None)
chip_114.columns = ["tss_chr", "tss_start", "tss_end", "tss_name", "tss_score", "tss_strand", "motif_chr",
                    "motif_start", "motif_end", "motif_score", "motif_id", "cell", "overlap"]
chip_114["tss_id"] = chip_114.apply(get_tss_id, axis=1)
chip_114.sample(5)


# In[24]:


fimo_114 = pd.read_table(fimo_114_f, sep="\t", header=None)
fimo_114.columns = ["tss_chr", "tss_start", "tss_end", "tss_name", "tss_score", "tss_strand", "motif_chr",
                    "motif_start", "motif_end", "motif_id", "motif_score", "motif_strand"]
fimo_114["tss_id"] = fimo_114.apply(get_tss_id, axis=1)
fimo_114.head()


# In[25]:


pool1_annot = pd.read_table(pool1_annot_f, sep="\t")
pool1_annot.head()


# ## 2. merge fimo/chip with tss tissue-sp values

# In[26]:


all_ts = all_ts[["Id", "tissue_sp_all", "tissue_sp_3"]]
all_ts.columns = ["tss_id", "tissue_spec_all_cage", "tissue_spec_three_cage"]
all_ts.sample(5)


# In[27]:


len(fimo)


# In[28]:


fimo_ts = fimo.merge(all_ts, on="tss_id")
print(len(fimo_ts))
fimo_ts.sample(5)


# In[29]:


missing_tss_ids_fimo = fimo[~fimo["tss_id"].isin(all_ts["tss_id"])]["tss_id"].unique()
len(missing_tss_ids_fimo)


# In[30]:


len(chip)


# In[31]:


chip_ts = chip.merge(all_ts, on="tss_id")
print(len(chip_ts))
chip_ts.sample(5)


# In[32]:


missing_tss_ids_chip = chip[~chip["tss_id"].isin(all_ts["tss_id"])]["tss_id"].unique()
len(missing_tss_ids_chip)


# In[33]:


fimo_114_ts = fimo_114.merge(all_ts, on="tss_id")
print(len(fimo_114_ts))
fimo_114_ts.sample(5)


# In[34]:


chip_114_ts = chip_114.merge(all_ts, on="tss_id")
print(len(chip_114_ts))
chip_114_ts.sample(5)


# ## 3. map tissue spec per TF

# In[36]:


tf_ts.columns = ["tf", "TF_tissue_spec_all", "TF_tissue_spec_three"]
tf_ts.head()


# In[37]:


chip_ts["motif_upper"] = chip_ts["motif_id"].str.upper()
chip_114_ts["motif_upper"] = chip_114_ts["motif_id"].str.upper()
chip_ts.head()


# In[38]:


chip_ts = chip_ts.merge(tf_ts, left_on="motif_upper", right_on="tf", how="left")
chip_114_ts = chip_114_ts.merge(tf_ts, left_on="motif_upper", right_on="tf", how="left")
chip_ts.head()


# In[39]:


fimo_ts["motif_upper"] = fimo_ts["motif_id"].str.upper()
fimo_114_ts["motif_upper"] = fimo_114_ts["motif_id"].str.upper()
fimo_ts.head()


# In[40]:


fimo_ts = fimo_ts.merge(tf_ts, left_on="motif_upper", right_on="tf", how="left")
fimo_114_ts = fimo_114_ts.merge(tf_ts, left_on="motif_upper", right_on="tf", how="left")
fimo_ts.head()


# ## 4. find avg tissue-specifity of TFs within a given gene
# ### for both chip and fimo, both all samples and 3 samples only

# #### chip -- all promoters -- 3kb -- all CAGE

# In[41]:


chip_ts_sub = chip_ts[(~pd.isnull(chip_ts["tissue_spec_all_cage"])) & (~pd.isnull(chip_ts["TF_tissue_spec_all"]))]


# In[42]:


chip_grp_all = chip_ts_sub.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
chip_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
chip_grp_all.head()


# In[43]:


len(chip_grp_all)


# #### fimo -- all promoters -- 3kb -- all CAGE

# In[44]:


fimo_ts_sub = fimo_ts[(~pd.isnull(fimo_ts["tissue_spec_all_cage"])) & (~pd.isnull(fimo_ts["TF_tissue_spec_all"]))]


# In[45]:


fimo_grp_all = fimo_ts_sub.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
fimo_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
fimo_grp_all.head()


# In[46]:


len(fimo_grp_all)


# #### chip -- all promoters -- 114bp -- all CAGE

# In[47]:


chip_114_ts_sub = chip_114_ts[(~pd.isnull(chip_114_ts["tissue_spec_all_cage"])) & (~pd.isnull(chip_114_ts["TF_tissue_spec_all"]))]


# In[48]:


chip_114_grp_all = chip_114_ts_sub.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
chip_114_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
chip_114_grp_all.head()


# In[49]:


len(chip_114_grp_all)


# #### fimo -- all promoters -- 114bp -- all CAGE

# In[50]:


fimo_114_ts_sub = fimo_114_ts[(~pd.isnull(fimo_114_ts["tissue_spec_all_cage"])) & (~pd.isnull(fimo_114_ts["TF_tissue_spec_all"]))]


# In[51]:


fimo_114_grp_all = fimo_114_ts_sub.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
fimo_114_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
fimo_114_grp_all.head()


# In[52]:


len(fimo_114_grp_all)


# #### chip -- all promoters -- 3kb -- 3 cell-line CAGE

# In[53]:


chip_ts_sub = chip_ts[(~pd.isnull(chip_ts["tissue_spec_three_cage"])) & (~pd.isnull(chip_ts["TF_tissue_spec_three"]))]


# In[54]:


chip_grp_3 = chip_ts_sub.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
chip_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
chip_grp_3.head()


# In[55]:


len(chip_grp_3)


# #### fimo -- all promoters -- 3kb -- 3 cell-line CAGE

# In[56]:


fimo_ts_sub = fimo_ts[(~pd.isnull(fimo_ts["tissue_spec_three_cage"])) & (~pd.isnull(fimo_ts["TF_tissue_spec_three"]))]


# In[57]:


fimo_grp_3 = fimo_ts_sub.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
fimo_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
fimo_grp_3.head()


# In[58]:


len(fimo_grp_3)


# #### chip -- all promoters -- 114bp -- 3 cell-line CAGE

# In[59]:


chip_114_ts_sub = chip_114_ts[(~pd.isnull(chip_114_ts["tissue_spec_three_cage"])) & (~pd.isnull(chip_114_ts["TF_tissue_spec_three"]))]


# In[60]:


chip_114_grp_3 = chip_114_ts_sub.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
chip_114_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
chip_114_grp_3.head()


# In[61]:


len(chip_114_grp_3)


# #### fimo -- all promoters -- 114bp -- 3 cell-line CAGE

# In[62]:


fimo_114_ts_sub = fimo_114_ts[(~pd.isnull(fimo_114_ts["tissue_spec_three_cage"])) & (~pd.isnull(fimo_114_ts["TF_tissue_spec_three"]))]


# In[63]:


fimo_114_grp_3 = fimo_114_ts_sub.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
fimo_114_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
fimo_114_grp_3.head()


# In[64]:


len(fimo_114_grp_3)


# ## 5. plot the scatters

# In[65]:


cmap = sns.light_palette("darkslategray", as_cmap=True)


# In[66]:


tf_ts.sort_values(by="TF_tissue_spec_all").tail()


# #### chip -- all promoters -- 3kb -- 3 cell-line CAGE

# In[72]:


# fig = plt.figure(figsize=(1.2, 1.2))
# ax = sns.kdeplot(chip_grp_all["tf_ts"], chip_grp_all["tss_ts"], cmap=cmap, 
#                  shade=True, shade_lowest=False)
# ax.set_xlabel("mean(tissue-specificity of TFs w/ peak in TSS)")
# ax.set_ylabel("tissue-specificity of TSS")

# r, p = stats.spearmanr(chip_grp_all["tf_ts"], chip_grp_all["tss_ts"])
# print("r: %s, spearman p: %s" % (r, p))
# ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[75]:


g = sns.jointplot(data=chip_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("chip_allTSS_3kb_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### chip -- all promoters -- 3kb -- all CAGE

# In[77]:


g = sns.jointplot(data=chip_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.05), ylim=(0.5,1.05), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("chip_allTSS_3kb_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### chip -- all promoters -- 114bp -- 3 cell-line CAGE

# In[79]:


g = sns.jointplot(data=chip_114_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("chip_allTSS_114bp_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### chip -- all promoters -- 114bp -- all CAGE

# In[81]:


g = sns.jointplot(data=chip_114_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.05), ylim=(0.5,1.05), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("chip_allTSS_114bp_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### fimo -- all promoters -- 3kb -- 3 cell-line CAGE

# In[72]:


# fig = plt.figure(figsize=(1.2, 1.2))
# ax = sns.kdeplot(chip_grp_all["tf_ts"], chip_grp_all["tss_ts"], cmap=cmap, 
#                  shade=True, shade_lowest=False)
# ax.set_xlabel("mean(tissue-specificity of TFs w/ peak in TSS)")
# ax.set_ylabel("tissue-specificity of TSS")

# r, p = stats.spearmanr(chip_grp_all["tf_ts"], chip_grp_all["tss_ts"])
# print("r: %s, spearman p: %s" % (r, p))
# ax.annotate("r = {:.2f}".format(r), xy=(.05, .9), xycoords=ax.transAxes, fontsize=fontsize)


# In[82]:


g = sns.jointplot(data=fimo_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("fimo_allTSS_3kb_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### fimo -- all promoters -- 3kb -- all CAGE

# In[83]:


g = sns.jointplot(data=fimo_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.05), ylim=(0.5,1.05), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("fimo_allTSS_3kb_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### fimo -- all promoters -- 114bp -- 3 cell-line CAGE

# In[84]:


g = sns.jointplot(data=fimo_114_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("fimo_allTSS_114bp_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# #### fimo -- all promoters -- 114bp -- all CAGE

# In[85]:


g = sns.jointplot(data=fimo_114_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.05), ylim=(0.5,1.05), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("fimo_allTSS_114bp_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# ## 10. limit to pool1 tss only

# In[86]:


fimo_filt = fimo_ts[fimo_ts["tss_id"].isin(pool1_annot["TSS_id"])]
fimo_filt.head()


# In[87]:


len(fimo_filt)


# In[88]:


fimo_114_filt = fimo_114_ts[fimo_114_ts["tss_id"].isin(pool1_annot["TSS_id"])]
len(fimo_114_filt)


# In[89]:


chip_filt = chip_ts[chip_ts["tss_id"].isin(pool1_annot["TSS_id"])]
len(chip_filt)


# In[90]:


chip_114_filt = chip_114_ts[chip_114_ts["tss_id"].isin(pool1_annot["TSS_id"])]
len(chip_114_filt)


# In[92]:


fimo_filt_grp_all = fimo_filt.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
fimo_filt_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[93]:


fimo_filt_grp_3 = fimo_filt.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
fimo_filt_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[94]:


fimo_114_filt_grp_all = fimo_114_filt.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
fimo_114_filt_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[95]:


fimo_114_filt_grp_3 = fimo_114_filt.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
fimo_114_filt_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[96]:


chip_filt_grp_all = chip_filt.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
chip_filt_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[97]:


chip_filt_grp_3 = chip_filt.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
chip_filt_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[98]:


chip_114_filt_grp_all = chip_114_filt.groupby(["tss_id", "tissue_spec_all_cage"])["TF_tissue_spec_all"].agg(["mean", "count"]).reset_index()
chip_114_filt_grp_all.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[99]:


chip_114_filt_grp_3 = chip_114_filt.groupby(["tss_id", "tissue_spec_three_cage"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
chip_114_filt_grp_3.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]


# In[101]:


g = sns.jointplot(data=chip_filt_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.1), ylim=(0.5,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("chip_pool1TSS_3kb_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[103]:


g = sns.jointplot(data=chip_filt_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("chip_pool1TSS_3kb_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[104]:


g = sns.jointplot(data=chip_114_filt_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.1), ylim=(0.5,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("chip_pool1TSS_114bp_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[105]:


g = sns.jointplot(data=chip_114_filt_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("chip_pool1TSS_114bp_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[106]:


g = sns.jointplot(data=fimo_filt_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.1), ylim=(0.5,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("fimo_pool1TSS_3kb_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[107]:


g = sns.jointplot(data=fimo_filt_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("fimo_pool1TSS_3kb_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[108]:


g = sns.jointplot(data=fimo_114_filt_grp_all, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0.5, 1.1), ylim=(0.5,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("fimo_pool1TSS_114bp_allCAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[109]:


g = sns.jointplot(data=fimo_114_filt_grp_3, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 0.8), ylim=(0,0.8), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "cell-type-specificity of TSS")
g.savefig("fimo_pool1TSS_114bp_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# ## 11. re-do MPRA tissue-sp

# In[110]:


coverage = pd.read_table("../../data/04__coverage/motif_coverage.txt", sep="\t")
coverage.head()


# In[112]:


pool1_fimo = pd.read_table("../../misc/05__fimo/pool1_fimo_map.txt", sep="\t")
pool1_fimo.head()


# In[113]:


pool1_fimo = pool1_fimo.merge(tf_ts, left_on="#pattern name", right_on="tf", how="left")
pool1_fimo.sample(5)


# In[114]:


pool1_fimo = pool1_fimo.merge(coverage, left_on="sequence name", right_on="unique_id", how="left")
pool1_fimo.head()


# In[116]:


pool1_fimo_grp = pool1_fimo.groupby(["unique_id", "MPRA_tissue_sp"])["TF_tissue_spec_three"].agg(["mean", "count"]).reset_index()
pool1_fimo_grp.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
pool1_fimo_grp.sample(5)


# In[117]:


g = sns.jointplot(data=pool1_fimo_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 1), ylim=(0,1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(cell-type-specificity of TFs w/ peak in TSS)", "MPRA cell-type-specificity of TSS")
g.savefig("MPRA_spec_fimo_pool1TSS_114bp_3CAGE_corr.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




