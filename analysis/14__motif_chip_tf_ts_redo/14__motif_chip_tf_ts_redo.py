
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

# ### for all promoters (3kb)

# In[4]:


chip_f = "chip_all.txt"


# In[5]:


fimo_f = "fimo_all_biotypes.txt"


# In[6]:


tss_ts_f = "hg19.cage_peak_phase1and2combined_counts.osc.tissue_specificity.txt"
enh_ts_f = "human_permissive_enhancers_phase_1_and_2_expression_count_matrix.tissue_specificity.txt"


# In[7]:


tss_ts_f = "hg19.cage_peak_phase1and2combined_counts.osc.tissue_specificity.txt"
enh_ts_f = "human_permissive_enhancers_phase_1_and_2_expression_count_matrix.tissue_specificity.txt"


# In[8]:


annot_f = "../../misc/00__tss_properties/TSS_FantomCat_all.TSSperENSG.txt"
fimo_map_f = "../../misc/04__jaspar_id_map/2018_03_09_gencode_jaspar_curated.txt"
chip_map_f = "ensembl_92_gene_id_to_name.txt"


# In[9]:


tf_ts_f = "gtex_tissue_specificity_tau.txt"


# ### for all promoters (114bp)

# In[10]:


chip_114_f = "chip_all_114.txt"


# In[11]:


fimo_114_f = "../../misc/05__fimo/TFmotifs__intersect_114bpTSS.uniq.txt"


# ### pool1 tss

# In[12]:


pool1_annot_f = "../../misc/00__tss_properties/TABLE_ALL_TSS_and_flipped.properties.PromType.txt"


# ## 1. import data

# In[13]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo = fimo[fimo["shuffled"] != "shuffled"]
fimo.head()


# In[14]:


chip = pd.read_table(chip_f, sep="\t")
chip.head()


# In[15]:


annot = pd.read_table(annot_f, sep="\t")
promtype2 = annot[["gene_id", "PromType2"]].drop_duplicates()
promtype2.head()


# In[16]:


fimo_map = pd.read_table(fimo_map_f, sep="\t")
fimo_map.head()


# In[17]:


chip_map = pd.read_table(chip_map_f, sep="\t")
chip_map.head()


# In[18]:


tf_ts = pd.read_table(tf_ts_f, sep="\t")
tf_ts.head()


# In[19]:


tss_ts = pd.read_table(tss_ts_f, sep="\t")
tss_ts.head()


# In[20]:


enh_ts = pd.read_table(enh_ts_f, sep="\t")
enh_ts.head()


# In[21]:


all_ts = tss_ts.append(enh_ts)


# In[22]:


chip_114 = pd.read_table(chip_114_f, sep="\t", header=None)
chip_114.columns = ["tss_chr", "tss_start", "tss_end", "tss_name", "tss_score", "tss_strand", "motif_chr",
                    "motif_start", "motif_end", "motif_score", "motif_id", "cell", "overlap"]
chip_114["tss_id"] = chip_114.apply(get_tss_id, axis=1)
chip_114.sample(5)


# In[23]:


fimo_114 = pd.read_table(fimo_114_f, sep="\t", header=None)
fimo_114.columns = ["tss_chr", "tss_start", "tss_end", "tss_name", "tss_score", "tss_strand", "motif_chr",
                    "motif_start", "motif_end", "motif_id", "motif_score", "motif_strand"]
fimo_114["tss_id"] = fimo_114.apply(get_tss_id, axis=1)
fimo_114.head()


# In[24]:


pool1_annot = pd.read_table(pool1_annot_f, sep="\t")
pool1_annot.head()


# ## 2. grab gene_ids for motif_names in chip/fimo 

# In[23]:


fimo_tfs = list(fimo["motif_id"].unique())
len(fimo_tfs)


# In[24]:


chip_tfs = list(chip["motif_id"].unique())
len(chip_tfs)


# In[25]:


manual_aliases = {"SIN3AK20": "SIN3A", "KAP1": "TRIM28", "SREBP1": "SREBF1", "ZZZ3": "AC118549.1", 
                  "RPC155": "POLR3A", "RDBP": "NELFE", "FAM48A": "SUPT20H", "MZF1(VAR.2)": "MZF1",
                  "RORA(VAR.2)": "RORA", "JUN(VAR.2)": "JUN", "JUND(VAR.2)": "JUND", "NKX2-5(VAR.2)": "NKX2-5",
                  "JDP2(VAR.2)": "JDP2", "NR2F6(VAR.2)": "NR2F6", "RARA(VAR.2)": "RARA", "TFAP2A(VAR.2)": "TFAP2A",
                  "TFAP2B(VAR.2)": "TFAP2B", "TFAP2B(VAR.3)": "TFAP2B", "TFAP2C(VAR.2)": "TFAP2C", 
                  "SREBF2(VAR.2)": "SREBF2", "SREBF1(VAR.2)": "SREBF1", "RARB(VAR.2)": "RARB", 
                  "RARG(VAR.2)": "RARG", "TFAP2A(VAR.3)": "TFAP2A", "TFAP2C(VAR.3)": "TFAP2C", "MIX-A": "MIXL1"}
manual_gene_ids = {"HSF1": "ENSG00000185122", "HNF1B": "ENSG00000275410", "KLF13": "ENSG00000169926",
                   "POU5F1": "ENSG00000204531", "SMARCB1": "ENSG00000099956", "RXRB": "ENSG00000204231"}


# In[26]:


def get_gene_id(motif_names, fimo_map, chip_map, manual_aliases, manual_gene_ids):
    gene_id_map = {}
    for motif in motif_names:
        motif = motif.upper()
        
        if "::" in motif:
            # fusion protein, continue
            gene_id_map[motif] = np.nan
            continue
        
        if motif in manual_gene_ids:
            gene_id_map[motif] = manual_gene_ids[motif]
            continue
            
        try:
            fimo_id = fimo_map[fimo_map["motif_name"] == motif]["Gene ID"].iloc[0]
        except:
            fimo_id = "none"
        try:
            chip_id = chip_map[chip_map["Gene name"] == motif]["Gene stable ID"].iloc[0]
        except:
            chip_id = "none"
        if fimo_id == "none" and chip_id == "none":
            try:
                alias = manual_aliases[motif]
            except:
                print("%s: no id, no alias" % motif)
                gene_id_map[motif] = np.nan
                continue
            try:
                real_id = chip_map[chip_map["Gene name"] == alias]["Gene stable ID"].iloc[0]
            except:
                print("%s: no id found for alias %s" % (motif, alias))
                gene_id_map[motif] = np.nan
        elif fimo_id != "none" and chip_id == "none":
            #print("found fimo id")
            gene_id_map[motif] = fimo_id
        elif fimo_id == "none" and chip_id != "none":
            #print("found chip id")
            gene_id_map[motif] = chip_id
        elif fimo_id != "none" and chip_id != "none":
            if fimo_id == chip_id:
                #print("found fimo/chip id that agrees")
                gene_id_map[motif] = fimo_id
            else:
                #print("%s: found fimo/chip id that disagree: %s, %s\n" % (motif, fimo_id, chip_id))
                real_id = manual_gene_ids[motif]
                gene_id_map[motif] = real_id
    return gene_id_map


# In[27]:


chip_id_map = get_gene_id(chip_tfs, fimo_map, chip_map, manual_aliases, manual_gene_ids)


# In[28]:


fimo_id_map = get_gene_id(fimo_tfs, fimo_map, chip_map, manual_aliases, manual_gene_ids)


# In[29]:


chip_id_map = pd.DataFrame.from_dict(chip_id_map, orient="index").reset_index()
chip_id_map.columns = ["motif_name", "gene_id"]
chip_id_map.head()


# In[30]:


fimo_id_map = pd.DataFrame.from_dict(fimo_id_map, orient="index").reset_index()
fimo_id_map.columns = ["motif_name", "gene_id"]
fimo_id_map.head()


# ## 3. merge fimo/chip with tss tissue-sp values
# why are some TSSs missing?

# In[31]:


len(fimo)


# In[32]:


fimo_ts = fimo.merge(all_ts, on="tss_id")
print(len(fimo_ts))
fimo_ts.sample(5)


# In[33]:


missing_tss_ids_fimo = fimo[~fimo["tss_id"].isin(all_ts["tss_id"])]["tss_id"].unique()
len(missing_tss_ids_fimo)


# In[34]:


len(chip)


# In[35]:


chip_ts = chip.merge(all_ts, on="tss_id")
print(len(chip_ts))
chip_ts.sample(5)


# In[36]:


missing_tss_ids_chip = chip[~chip["tss_id"].isin(all_ts["tss_id"])]["tss_id"].unique()
len(missing_tss_ids_chip)


# In[37]:


fimo_114_ts = fimo_114.merge(all_ts, on="tss_id")
print(len(fimo_114_ts))
fimo_114_ts.sample(5)


# In[38]:


chip_114_ts = chip_114.merge(all_ts, on="tss_id")
print(len(chip_114_ts))
chip_114_ts.sample(5)


# ## 4. find tissue-sp per TF
# #### use gtex for now

# In[39]:


tf_ts["gene_id"] = tf_ts["GeneID"].str.split(".", expand=True)[0]
tf_ts.head()


# In[40]:


chip_id_map_ts = chip_id_map.merge(tf_ts, on="gene_id", how="left")
chip_id_map_ts.sample(5)


# In[41]:


chip_id_map_ts[pd.isnull(chip_id_map_ts["tissue_spec"])]


# In[42]:


fimo_id_map_ts = fimo_id_map.merge(tf_ts, on="gene_id", how="left")
fimo_id_map_ts.sample(5)


# In[43]:


fimo_id_map_ts[pd.isnull(fimo_id_map_ts["tissue_spec"])]


# ## 5. merge fimo/chip with tf spec values

# In[44]:


chip_ts["motif_upper"] = chip_ts["motif_id"].str.upper()
chip_114_ts["motif_upper"] = chip_114_ts["motif_id"].str.upper()
chip_ts.head()


# In[45]:


chip_ts = chip_ts.merge(chip_id_map_ts, left_on="motif_upper", right_on="motif_name", how="left")
chip_114_ts = chip_114_ts.merge(chip_id_map_ts, left_on="motif_upper", right_on="motif_name", how="left")
chip_ts.head()


# In[46]:


fimo_ts["motif_upper"] = fimo_ts["motif_id"].str.upper()
fimo_114_ts["motif_upper"] = fimo_114_ts["motif_id"].str.upper()
fimo_ts.head()


# In[47]:


fimo_ts = fimo_ts.merge(fimo_id_map_ts, left_on="motif_upper", right_on="motif_name", how="left")
fimo_114_ts = fimo_114_ts.merge(fimo_id_map_ts, left_on="motif_upper", right_on="motif_name", how="left")
fimo_ts.head()


# ## 6. find avg. tissue-spec for genes containing a given chip peak/motif

# In[48]:


chip_grp = chip_ts.groupby(["motif_id", "tissue_spec_tau"])["tissue_spec_x"].agg(["mean", "count"]).reset_index()
chip_grp.columns = ["motif_id", "tf_ts", "tss_ts", "tss_count"]
chip_grp.head()


# In[49]:


len(chip_grp)


# In[50]:


fimo_grp = fimo_ts.groupby(["motif_id", "tissue_spec_tau"])["tissue_spec_x"].agg(["mean", "count"]).reset_index()
fimo_grp.columns = ["motif_id", "tf_ts", "tss_ts", "tss_count"]
fimo_grp.head()


# In[51]:


len(fimo_grp)


# In[52]:


chip_114_grp = chip_114_ts.groupby(["motif_id", "tissue_spec_tau"])["tissue_spec_x"].agg(["mean", "count"]).reset_index()
chip_114_grp.columns = ["motif_id", "tf_ts", "tss_ts", "tss_count"]
chip_114_grp.head()


# In[53]:


len(chip_114_grp)


# In[54]:


fimo_114_grp = fimo_114_ts.groupby(["motif_id", "tissue_spec_tau"])["tissue_spec_x"].agg(["mean", "count"]).reset_index()
fimo_114_grp.columns = ["motif_id", "tf_ts", "tss_ts", "tss_count"]
fimo_114_grp.head()


# In[55]:


len(fimo_114_grp)


# ## 7. plot correlations

# In[56]:


g = sns.jointplot(data=chip_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray")
g.set_axis_labels("tissue-specificity of TF", "mean(tissue-specificity of TSSs w/ peak)")
g.savefig("chip_corr.pdf", dpi="figure", bbox_inches="tight")


# In[57]:


g = sns.jointplot(data=chip_114_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray")
g.set_axis_labels("tissue-specificity of TF", "mean(tissue-specificity of TSSs w/ peak)")
g.savefig("chip_corr_114.pdf", dpi="figure", bbox_inches="tight")


# In[58]:


g = sns.jointplot(data=fimo_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray")
g.set_axis_labels("tissue-specificity of TF", "mean(tissue-specificity of TSSs w/ motif)")
g.savefig("motif_corr.pdf", dpi="figure", bbox_inches="tight")


# In[59]:


g = sns.jointplot(data=fimo_114_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray")
g.set_axis_labels("tissue-specificity of TF", "mean(tissue-specificity of TSSs w/ motif)")
g.savefig("motif_corr_114.pdf", dpi="figure", bbox_inches="tight")


# ## 8. do the reverse: find avg. tissue-specifity of TFs within a given gene

# In[60]:


chip_grp_rev = chip_ts.groupby(["tss_id", "tissue_spec_x"])["tissue_spec_tau"].agg(["mean", "count"]).reset_index()
chip_grp_rev.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
chip_grp_rev.head()


# In[61]:


len(chip_grp_rev)


# In[62]:


fimo_grp_rev = fimo_ts.groupby(["tss_id", "tissue_spec_x"])["tissue_spec_tau"].agg(["mean", "count"]).reset_index()
fimo_grp_rev.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
fimo_grp_rev.head()


# In[63]:


len(fimo_grp_rev)


# In[64]:


chip_114_grp_rev = chip_114_ts.groupby(["tss_id", "tissue_spec_x"])["tissue_spec_tau"].agg(["mean", "count"]).reset_index()
chip_114_grp_rev.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
chip_114_grp_rev.head()


# In[65]:


len(chip_114_grp_rev)


# In[66]:


fimo_114_grp_rev = fimo_114_ts.groupby(["tss_id", "tissue_spec_x"])["tissue_spec_tau"].agg(["mean", "count"]).reset_index()
fimo_114_grp_rev.columns = ["tss_id", "tss_ts", "tf_ts", "tf_count"]
fimo_114_grp_rev.head()


# In[67]:


len(fimo_114_grp_rev)


# ## 9. plot the reverse

# In[68]:


chip_grp_rev["log_tss_ts"] = np.log(chip_grp_rev["tss_ts"]+1)
chip_grp_rev["log_tf_ts"] = np.log(chip_grp_rev["tf_ts"]+1)

fimo_grp_rev["log_tss_ts"] = np.log(fimo_grp_rev["tss_ts"]+1)
fimo_grp_rev["log_tf_ts"] = np.log(fimo_grp_rev["tf_ts"]+1)


# In[69]:


g = sns.jointplot(data=chip_grp_rev, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 1.1), ylim=(0,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("chip_corr_rev.pdf", dpi="figure", bbox_inches="tight")


# In[70]:


g = sns.jointplot(data=chip_114_grp_rev, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", 
                  stat_func=spearmanr,
                  xlim=(0, 1.1), ylim=(0,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ peak in TSS)", "tissue-specificity of TSS")
g.savefig("chip_corr_rev_114.pdf", dpi="figure", bbox_inches="tight")


# In[71]:


g = sns.jointplot(data=fimo_grp_rev, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", stat_func=spearmanr,
                  xlim=(0, 1.1), ylim=(0,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ motif in TSS)", "tissue-specificity of TSS")
g.savefig("fimo_corr_rev.pdf", dpi="figure", bbox_inches="tight")


# In[72]:


g = sns.jointplot(data=fimo_114_grp_rev, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray", stat_func=spearmanr,
                  xlim=(0, 1.1), ylim=(0,1.1), joint_kws=dict(scatter_kws={'alpha':0.1}))
g.set_axis_labels("mean(tissue-specificity of TFs w/ motif in TSS)", "tissue-specificity of TSS")
g.savefig("fimo_corr_rev_114.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




