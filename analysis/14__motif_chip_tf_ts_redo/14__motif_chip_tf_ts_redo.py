
# coding: utf-8

# # 13__motif_chip_tf_ts_redo
# # testing whether gene tissue-sp correlates with chip/motif tissue-sp
# 
# after talking to lucas: find tissue-sp for each TF, find all genes that have that motif/chip peak, calc. average tissue-sp. of those genes, correlate that with the tissue-sp. (and subset by class, if needed)
# 
# run outside of notebook:
#     ````bedtools intersect -wo -a ../../data/00__index/0__all_tss/TSS.start.1perGencGene.geneIdOnly.3kb.bed -b ../../misc/07__chip/ALL_CELL_LINES__all_ChIP_peaks.bed````
# also for enhancers
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


# ## variables

# In[3]:


chip_f = "chip_all.txt"


# In[4]:


fimo_f = "fimo_all_biotypes.txt"


# In[5]:


tss_ts_f = "hg19.cage_peak_phase1and2combined_counts.osc.tissue_specificity.txt"
enh_ts_f = "human_permissive_enhancers_phase_1_and_2_expression_count_matrix.tissue_specificity.txt"


# In[6]:


annot_f = "../../misc/00__tss_properties/TSS_FantomCat_all.TSSperENSG.txt"
fimo_map_f = "../../misc/04__jaspar_id_map/2018_03_09_gencode_jaspar_curated.txt"
chip_map_f = "ensembl_92_gene_id_to_name.txt"


# In[7]:


tf_ts_f = "gtex_tissue_specificity_tau.txt"


# ## 1. import data

# In[8]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo = fimo[fimo["shuffled"] != "shuffled"]
fimo.head()


# In[9]:


chip = pd.read_table(chip_f, sep="\t")
chip.head()


# In[10]:


tss_ts = pd.read_table(tss_ts_f, sep="\t")
tss_ts.head()


# In[11]:


enh_ts = pd.read_table(enh_ts_f, sep="\t")
enh_ts.head()


# In[12]:


all_ts = tss_ts.append(enh_ts)


# In[13]:


annot = pd.read_table(annot_f, sep="\t")
promtype2 = annot[["gene_id", "PromType2"]].drop_duplicates()
promtype2.head()


# In[14]:


fimo_map = pd.read_table(fimo_map_f, sep="\t")
fimo_map.head()


# In[15]:


chip_map = pd.read_table(chip_map_f, sep="\t")
chip_map.head()


# In[16]:


tf_ts = pd.read_table(tf_ts_f, sep="\t")
tf_ts.head()


# ## 2. grab gene_ids for motif_names in chip/fimo 

# In[17]:


fimo_tfs = list(fimo["motif_id"].unique())
len(fimo_tfs)


# In[18]:


chip_tfs = list(chip["motif_id"].unique())
len(chip_tfs)


# In[19]:


manual_aliases = {"SIN3AK20": "SIN3A", "KAP1": "TRIM28", "SREBP1": "SREBF1", "ZZZ3": "AC118549.1", 
                  "RPC155": "POLR3A", "RDBP": "NELFE", "FAM48A": "SUPT20H", "MZF1(VAR.2)": "MZF1",
                  "RORA(VAR.2)": "RORA", "JUN(VAR.2)": "JUN", "JUND(VAR.2)": "JUND", "NKX2-5(VAR.2)": "NKX2-5",
                  "JDP2(VAR.2)": "JDP2", "NR2F6(VAR.2)": "NR2F6", "RARA(VAR.2)": "RARA", "TFAP2A(VAR.2)": "TFAP2A",
                  "TFAP2B(VAR.2)": "TFAP2B", "TFAP2B(VAR.3)": "TFAP2B", "TFAP2C(VAR.2)": "TFAP2C", 
                  "SREBF2(VAR.2)": "SREBF2", "SREBF1(VAR.2)": "SREBF1", "RARB(VAR.2)": "RARB", 
                  "RARG(VAR.2)": "RARG", "TFAP2A(VAR.3)": "TFAP2A", "TFAP2C(VAR.3)": "TFAP2C", "MIX-A": "MIXL1"}
manual_gene_ids = {"HSF1": "ENSG00000185122", "HNF1B": "ENSG00000275410", "KLF13": "ENSG00000169926",
                   "POU5F1": "ENSG00000204531", "SMARCB1": "ENSG00000099956", "RXRB": "ENSG00000204231"}


# In[20]:


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


# In[21]:


chip_id_map = get_gene_id(chip_tfs, fimo_map, chip_map, manual_aliases, manual_gene_ids)


# In[22]:


fimo_id_map = get_gene_id(fimo_tfs, fimo_map, chip_map, manual_aliases, manual_gene_ids)


# In[23]:


chip_id_map = pd.DataFrame.from_dict(chip_id_map, orient="index").reset_index()
chip_id_map.columns = ["motif_name", "gene_id"]
chip_id_map.head()


# In[24]:


fimo_id_map = pd.DataFrame.from_dict(fimo_id_map, orient="index").reset_index()
fimo_id_map.columns = ["motif_name", "gene_id"]
fimo_id_map.head()


# ## 3. merge fimo/chip with tss tissue-sp values
# why are some TSSs missing?

# In[25]:


len(fimo)


# In[26]:


fimo_ts = fimo.merge(all_ts, on="tss_id")
print(len(fimo_ts))
fimo_ts.sample(5)


# In[27]:


missing_tss_ids_fimo = fimo[~fimo["tss_id"].isin(fimo_ts["tss_id"])]["tss_id"].unique()
len(missing_tss_ids_fimo)


# In[28]:


len(chip)


# In[29]:


chip_ts = chip.merge(all_ts, on="tss_id")
print(len(chip_ts))
chip_ts.sample(5)


# In[30]:


missing_tss_ids_chip = chip[~chip["tss_id"].isin(chip_ts["tss_id"])]["tss_id"].unique()
len(missing_tss_ids_chip)


# ## 4. find tissue-sp per TF
# #### use gtex for now

# In[31]:


tf_ts["gene_id"] = tf_ts["GeneID"].str.split(".", expand=True)[0]
tf_ts.head()


# In[32]:


chip_id_map_ts = chip_id_map.merge(tf_ts, on="gene_id", how="left")
chip_id_map_ts.sample(5)


# In[33]:


chip_id_map_ts[pd.isnull(chip_id_map_ts["tissue_spec"])]


# In[34]:


fimo_id_map_ts = fimo_id_map.merge(tf_ts, on="gene_id", how="left")
fimo_id_map_ts.sample(5)


# In[35]:


fimo_id_map_ts[pd.isnull(fimo_id_map_ts["tissue_spec"])]


# ## 5. merge fimo/chip with tf spec values

# In[37]:


chip_ts["motif_upper"] = chip_ts["motif_id"].str.upper()
chip_ts.head()


# In[38]:


chip_ts = chip_ts.merge(chip_id_map_ts, left_on="motif_upper", right_on="motif_name", how="left")
chip_ts.head()


# In[39]:


fimo_ts["motif_upper"] = fimo_ts["motif_id"].str.upper()
fimo_ts.head()


# In[40]:


fimo_ts = fimo_ts.merge(fimo_id_map_ts, left_on="motif_upper", right_on="motif_name", how="left")
fimo_ts.head()


# ## 6. find avg. tissue-spec for genes containing a given chip peak/motif

# In[60]:


chip_grp = chip_ts.groupby(["motif_id", "tissue_spec_tau"])["tissue_spec_x"].agg(["mean", "count"]).reset_index()
chip_grp.columns = ["motif_id", "tf_ts", "tss_ts", "tss_count"]
chip_grp.head()


# In[61]:


len(chip_grp)


# In[62]:


fimo_grp = fimo_ts.groupby(["motif_id", "tissue_spec_tau"])["tissue_spec_x"].agg(["mean", "count"]).reset_index()
fimo_grp.columns = ["motif_id", "tf_ts", "tss_ts", "tss_count"]
fimo_grp.head()


# In[63]:


len(fimo_grp)


# ## 7. plot correlations

# In[74]:


g = sns.jointplot(data=chip_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray")
g.set_axis_labels("tissue-specificity of TF", "mean(tissue-specificity of TSSs w/ peak)")
g.savefig("chip_corr.pdf", dpi="figure", bbox_inches="tight")


# In[75]:


g = sns.jointplot(data=fimo_grp, x="tf_ts", y="tss_ts", kind="reg", size=2.5, color="gray")
g.set_axis_labels("tissue-specificity of TF", "mean(tissue-specificity of TSSs w/ motif)")
g.savefig("motif_corr.pdf", dpi="figure", bbox_inches="tight")


# In[ ]:




