
# coding: utf-8

# # 13__fimo_v_mind
# # comparing ChIP overlap b/w FIMO and MIND
# 
# in this notebook, i compare the accuracy of FIMO and MIND by comparing how often hits from either one overlap with known ChIP-seq peaks. 

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


chip_dir = "../../misc/07__chip"


# In[4]:


hepg2_motif_dir = "../../data/06__mind_results/HepG2/files/1__motif_scores"
k562_motif_dir = "../../data/06__mind_results/K562/files/1__motif_scores"


# In[5]:


fimo_f = "../../misc/05__fimo/pool2.fimo.txt"


# In[6]:


index_f = "../../data/00__index/dels_oligo_pool.index.txt"


# In[7]:


del_buffer = 10
fdr_cutoff = 0.05


# ## 1. import data

# In[8]:


hepg2_files = []
for (dirpath, dirnames, filenames) in walk(hepg2_motif_dir):
    hepg2_files.extend(filenames)
    break
hepg2_files = [x for x in hepg2_files if "loss__results" in x and "expr_filt" not in x]
hepg2_files[0:5]


# In[9]:


k562_files = []
for (dirpath, dirnames, filenames) in walk(k562_motif_dir):
    k562_files.extend(filenames)
    break
k562_files = [x for x in k562_files if "loss__results" in x and "expr_filt" not in x]
k562_files[0:5]


# In[10]:


fimo = pd.read_table(fimo_f, sep="\t")
fimo.head()


# In[11]:


index = pd.read_table(index_f, sep="\t")
index.head()


# ## 2. find motifs < 0.05 FDR & put in terms of absolute coords

# In[12]:


hepg2_data = {}
for f in hepg2_files:
    name = f.split(".")[0]
    df = pd.read_table("%s/%s" % (hepg2_motif_dir, f), sep="\t")
    hepg2_data[name] = df
    #print("%s: min motif nuc: %s, max motif nuc: %s" % (name, df["start"].min(), df["end"].max()))
hepg2_data["GAS5__p1__tile2__minus__peak1"].head()


# In[13]:


k562_data = {}
for f in k562_files:
    name = f.split(".")[0]
    df = pd.read_table("%s/%s" % (k562_motif_dir, f), sep="\t")
    k562_data[name] = df
    #print("%s: min motif nuc: %s, max motif nuc: %s" % (name, df["start"].min(), df["end"].max()))
k562_data["GAS5__p1__tile2__minus__peak1"].head()


# In[14]:


def get_strand(row):
    if row["strand"] == "sense":
        return "+"
    else:
        return "-"


# In[15]:


hepg2_all_motifs = pd.DataFrame()
hepg2_top5_motifs = pd.DataFrame()
for name in hepg2_data:
    df = hepg2_data[name]

    if "plus" in name:
        df["motif_start"] = df["tile_start"].astype(int) + del_buffer + df["start"].astype(int) - 1
        df["motif_end"] = df["tile_start"].astype(int) + del_buffer + df["start"].astype(int) + (df["end"]-df["start"]).astype(int) - 1
    elif "minus" in name:
        df["motif_end"] = df["tile_end"].astype(int) - del_buffer - df["start"].astype(int) + 1
        df["motif_start"] = df["motif_end"].astype(int) - (df["end"]-df["start"]).astype(int)
        
    # limit to motifs < fdr cutoff
    fdr = df["fdr_cutoff"].iloc[0]
    sig_motifs = df[df["padj"] < fdr]
    sig_motifs["gene"] = name
    sig_motifs["id"] = sig_motifs["motif"].astype(str) + "__" + sig_motifs["tile_chr"].astype(str) + "__" + sig_motifs["motif_start"].astype(str)
    
    # for other analysis - limit to top 5
    top5_motifs = sig_motifs.sort_values(by="score", ascending=False).head(5)
    
    hepg2_all_motifs = hepg2_all_motifs.append(sig_motifs)
    hepg2_top5_motifs = hepg2_top5_motifs.append(top5_motifs)

hepg2_all_motifs["strand"] = hepg2_all_motifs.apply(get_strand, axis=1)
hepg2_all_motifs = hepg2_all_motifs[["tile_chr", "motif_start", "motif_end", "motif", "score", "strand", "gene", "id"]]
hepg2_all_motifs = hepg2_all_motifs.drop_duplicates()
hepg2_all_motifs.sample(5)


# In[16]:


k562_all_motifs = pd.DataFrame()
k562_top5_motifs = pd.DataFrame()

for name in k562_data:
    df = k562_data[name]

    if "plus" in name:
        df["motif_start"] = df["tile_start"].astype(int) + del_buffer + df["start"].astype(int) - 1
        df["motif_end"] = df["tile_start"].astype(int) + del_buffer + df["start"].astype(int) + (df["end"]-df["start"]).astype(int) - 1
    elif "minus" in name:
        df["motif_end"] = df["tile_end"].astype(int) - del_buffer - df["start"].astype(int) + 1
        df["motif_start"] = df["motif_end"].astype(int) - (df["end"]-df["start"]).astype(int)
        
    # limit to motifs < 0.05
    fdr = df["fdr_cutoff"].iloc[0]
    sig_motifs = df[df["padj"] < fdr]
    sig_motifs["gene"] = name
    sig_motifs["id"] = sig_motifs["motif"].astype(str) + "__" + sig_motifs["tile_chr"].astype(str) + "__" + sig_motifs["motif_start"].astype(str)
    
    # for other analysis - limit to top 5
    top5_motifs = sig_motifs.sort_values(by="score", ascending=False).head(5)
    
    k562_all_motifs = k562_all_motifs.append(sig_motifs)
    k562_top5_motifs = k562_top5_motifs.append(top5_motifs)

k562_all_motifs["strand"] = k562_all_motifs.apply(get_strand, axis=1)
k562_all_motifs = k562_all_motifs[["tile_chr", "motif_start", "motif_end", "motif", "score", "strand", "gene", "id"]]
k562_all_motifs = k562_all_motifs.drop_duplicates()
k562_all_motifs.sample(5)


# In[17]:


hepg2_all_motifs["cell_type"] = "HepG2"
k562_all_motifs["cell_type"] = "K562"
all_motifs = hepg2_all_motifs.append(k562_all_motifs)


# In[18]:


len(all_motifs)


# ## same for fimo

# In[19]:


fimo_wt = fimo[~(fimo["sequence name"].str.contains("DELETION")) & 
               ~(fimo["sequence name"].str.contains("SNP_INDIV"))]
fimo_wt.sample(5)


# In[20]:


fimo_wt = fimo_wt.merge(index[["unique_id", "chr", "tile_start", "tile_end"]].drop_duplicates(), 
                        left_on="sequence name", right_on="unique_id")
fimo_wt.sample(5)


# In[21]:


fimo_wt["motif_start"] = fimo_wt["tile_start"].astype(int) + fimo_wt["start"].astype(int) - 1
fimo_wt["motif_end"] = fimo_wt["tile_start"].astype(int) + fimo_wt["stop"].astype(int)
fimo_wt["id"] = fimo_wt["#pattern name"] + "__" + fimo_wt["chr"] + "__" + fimo_wt["motif_start"].astype(str)
fimo_wt.sample(5)


# In[22]:


fimo_bed = fimo_wt[["chr", "motif_start", "motif_end", "#pattern name", "score", "strand", "sequence name", "id"]]
fimo_bed.columns = ["tile_chr", "motif_start", "motif_end", "motif", "score", "strand", "gene", "id"]
fimo_bed.head()


# In[23]:


len(fimo_bed)


# In[24]:


# make df w/ a cell column
# since fimo = same across cell types, obviously, just repeat df
hepg2_fimo = fimo_bed.copy()
hepg2_fimo["cell_type"] = "HepG2"
k562_fimo = fimo_bed.copy()
k562_fimo["cell_type"] = "K562"
all_fimo = hepg2_fimo.append(k562_fimo)


# ## 3. intersect w/ bed files for cell type

# In[25]:


chip_files = []
for (dirpath, dirnames, filenames) in walk(chip_dir):
    chip_files.extend(filenames)
    break

chip_files = [x for x in chip_files if "__" in x]
hepg2_chip_files = [x for x in chip_files if "HepG2" in x]
k562_chip_files = [x for x in chip_files if "K562" in x]
k562_chip_files[0:5]


# In[26]:


print("total chip files: %s" % (len(chip_files)))
print("total HepG2 chip files: %s" % (len(hepg2_chip_files)))
print("total K562 chip files: %s" % (len(k562_chip_files)))


# In[27]:


hepg2_chip_motifs = [x.split("__")[1].upper() for x in hepg2_chip_files]
k562_chip_motifs = [x.split("__")[1].upper() for x in k562_chip_files]
chip_motifs = [x.split("__")[1].upper() for x in chip_files]
chip_motifs[0:5]


# In[28]:


uniq_motifs = sorted(list(all_motifs["motif"].unique()))
len(uniq_motifs)


# In[29]:


def get_overlap(row, int_ids):
    if row["id"] in int_ids:
        return "ChIP_overlap"
    else:
        return "no_overlap"


# In[30]:


def intersect_chip_files(sub_df, chip_filename, colname):
    sub_df.to_csv("tmp.motifs.bed", sep="\t", index=False, header=False)
    get_ipython().system('bedtools intersect -u -a tmp.motifs.bed -b $chip_filename > tmp.txt')
    
    try:
        results = pd.read_table("tmp.txt", sep="\t", header=None)
        int_ids = list(results[7])
        sub_df[colname] = sub_df.apply(get_overlap, int_ids=int_ids, axis=1)
    except:
        sub_df[colname] = "no_overlap"
    return sub_df


# In[31]:


def merge_and_intersect_chip_files(chip_files, motif, chip_dir, sub_df, colname):
    chip_files_sub = [x for x in chip_files if motif in x]
    chip_paths = ["%s/%s" % (chip_dir, x) for x in chip_files_sub]
    all_chip = pd.DataFrame()
    for f in chip_paths:
        tmp = pd.read_table(f, sep="\t", header=None)
        all_chip = all_chip.append(tmp)
    all_chip.to_csv("tmp.chip.bed", sep="\t", header=False, index=False)

    # intersect bed files
    sub_df = intersect_chip_files(sub_df, "tmp.chip.bed", colname)
    return sub_df


# In[32]:


all_motifs_int = pd.DataFrame()
n_chip = 0
n_cell_chip = 0

for motif in uniq_motifs:
    print("=== %s ===" % motif)
    sub_df = all_motifs[all_motifs["motif"] == motif]
    
    # first check if these motifs overlap *any* chip peaks
    # sub_df = intersect_chip_files(sub_df, "%s/all.chip.bed.gz" % chip_dir, "any_chip_overlap")
    
    # force motif to upper
    motif_up = motif.upper()
    
    # see if that motif has been chipped at all
    if motif_up in chip_motifs:
        print("has been chipped")
        n_chip += 1

        sub_df = merge_and_intersect_chip_files(chip_files, motif_up, chip_dir, sub_df, "all_chip_overlap")
            
        # do both k562 & hepg2
        for cell, cell_chip_motifs, cell_chip_files in zip(["HepG2", "K562"], [hepg2_chip_motifs, k562_chip_motifs], [hepg2_chip_files, k562_chip_files]):
            cell_df = sub_df[sub_df["cell_type"] == cell]

            # now see if that motif has been chipped in cell type
            if motif_up in cell_chip_motifs:
                print("in %s" % cell)
                n_cell_chip += 1

                # find chip files and merge again
                cell_df = merge_and_intersect_chip_files(cell_chip_files, motif_up, chip_dir, cell_df, 
                                                        "cell_chip_overlap")
            else:
                cell_df["cell_chip_overlap"] = "no_ChIP"
            
            all_motifs_int = all_motifs_int.append(cell_df)
    else:
        sub_df["all_chip_overlap"] = "no_ChIP"
        sub_df["cell_chip_overlap"] = "no_ChIP"
        all_motifs_int = all_motifs_int.append(sub_df)


# In[33]:


all_motifs_int[all_motifs_int["motif"] == "NRF1"]


# ## same for fimo

# In[34]:


fimo_uniq_motifs = sorted(list(all_fimo["motif"].unique()))
len(fimo_uniq_motifs)


# In[35]:


fimo_motifs_int = pd.DataFrame()
n_chip = 0
n_cell_chip = 0

for motif in fimo_uniq_motifs:
    print("=== %s ===" % motif)
    sub_df = all_fimo[all_fimo["motif"] == motif]
    
    # first check if these motifs overlap *any* chip peaks
    # sub_df = intersect_chip_files(sub_df, "%s/all.chip.bed.gz" % chip_dir, "any_chip_overlap")
    
    # force motif to upper
    motif_up = motif.upper()
    
    # see if that motif has been chipped at all
    if motif_up in chip_motifs:
        print("has been chipped")
        n_chip += 1

        sub_df = merge_and_intersect_chip_files(chip_files, motif_up, chip_dir, sub_df, "all_chip_overlap")
            
        # do both k562 & hepg2
        for cell, cell_chip_motifs, cell_chip_files in zip(["HepG2", "K562"], [hepg2_chip_motifs, k562_chip_motifs], [hepg2_chip_files, k562_chip_files]):
            cell_df = sub_df[sub_df["cell_type"] == cell]
            
            # now see if that motif has been chipped in cell type
            if motif_up in cell_chip_motifs:
                print("in %s" % cell)
                n_cell_chip += 1

                # find chip files and merge again
                cell_df = merge_and_intersect_chip_files(cell_chip_files, motif_up, chip_dir, cell_df, 
                                                         "cell_chip_overlap")
            else:
                cell_df["cell_chip_overlap"] = "no_ChIP"
            
            fimo_motifs_int = fimo_motifs_int.append(cell_df)
    else:
        sub_df["all_chip_overlap"] = "no_ChIP"
        sub_df["cell_chip_overlap"] = "no_ChIP"
        fimo_motifs_int = fimo_motifs_int.append(sub_df)


# In[36]:


fimo_motifs_int[fimo_motifs_int["motif"] == "NRF1"].head()


# ## 4. find numbers

# In[37]:


cell_results = pd.pivot_table(all_motifs_int, values="id", index="cell_chip_overlap", columns="cell_type",
                              aggfunc="count").reset_index()
cell_results


# In[38]:


all_results = pd.pivot_table(all_motifs_int, values="id", index="all_chip_overlap", columns="cell_type",
                             aggfunc="count").reset_index()
all_results


# In[39]:


cell_results_melt = pd.melt(cell_results, id_vars="cell_chip_overlap")
all_results_melt = pd.melt(all_results, id_vars="all_chip_overlap")

cell_results_melt = cell_results_melt[cell_results_melt["cell_chip_overlap"] != "no_ChIP"]
all_results_melt = all_results_melt[all_results_melt["all_chip_overlap"] != "no_ChIP"]

all_results_melt.head()


# In[40]:


cell_results_counts = cell_results_melt.groupby("cell_type")["value"].agg("sum").reset_index()
cell_results_melt = cell_results_melt.merge(cell_results_counts, on="cell_type", suffixes=("_count", "_total"))
cell_results_melt["percent"] = (cell_results_melt["value_count"]/cell_results_melt["value_total"])*100

all_results_counts = all_results_melt.groupby("cell_type")["value"].agg("sum").reset_index()
all_results_melt = all_results_melt.merge(all_results_counts, on="cell_type", suffixes=("_count", "_total"))
all_results_melt["percent"] = (all_results_melt["value_count"]/all_results_melt["value_total"])*100

all_results_melt.head()


# In[41]:


cell_results_melt = cell_results_melt[cell_results_melt["cell_chip_overlap"] == "ChIP_overlap"]
cell_results_melt["type"] = "cell-type ChIP"
cell_results_melt = cell_results_melt[["type", "cell_type", "percent"]]

all_results_melt = all_results_melt[all_results_melt["all_chip_overlap"] == "ChIP_overlap"]
all_results_melt["type"] = "all ChIP"
all_results_melt = all_results_melt[["type", "cell_type", "percent"]]

results_melt = cell_results_melt.append(all_results_melt)
results_melt


# In[42]:


figs_dir = "figs/0__overlap_bar"
get_ipython().system('mkdir -p $figs_dir')

plt.figure(figsize=(1.8, 1.9))
sns.barplot(data=results_melt[results_melt["type"] == "all ChIP"], 
            x="cell_type", y="percent", color=sns.color_palette("Set2")[0])
plt.xlabel("")
plt.ylabel("% of motifs that overlap a ChIP peak")
plt.ylim((0, 100))
plt.savefig("%s/all_chip_overlap.pdf" % figs_dir, dpi="figure", bbox_inches="tight")


# ## same with fimo

# In[43]:


cell_results = pd.pivot_table(fimo_motifs_int, values="id", index="cell_chip_overlap", columns="cell_type",
                              aggfunc="count").reset_index()
cell_results


# In[44]:


all_results = pd.pivot_table(fimo_motifs_int, values="id", index="all_chip_overlap", columns="cell_type",
                             aggfunc="count").reset_index()
all_results


# In[45]:


cell_results_melt = pd.melt(cell_results, id_vars="cell_chip_overlap")
all_results_melt = pd.melt(all_results, id_vars="all_chip_overlap")

cell_results_melt = cell_results_melt[cell_results_melt["cell_chip_overlap"] != "no_ChIP"]
all_results_melt = all_results_melt[all_results_melt["all_chip_overlap"] != "no_ChIP"]

all_results_melt.head()


# In[46]:


cell_results_counts = cell_results_melt.groupby("cell_type")["value"].agg("sum").reset_index()
cell_results_melt = cell_results_melt.merge(cell_results_counts, on="cell_type", suffixes=("_count", "_total"))
cell_results_melt["percent"] = (cell_results_melt["value_count"]/cell_results_melt["value_total"])*100

all_results_counts = all_results_melt.groupby("cell_type")["value"].agg("sum").reset_index()
all_results_melt = all_results_melt.merge(all_results_counts, on="cell_type", suffixes=("_count", "_total"))
all_results_melt["percent"] = (all_results_melt["value_count"]/all_results_melt["value_total"])*100

all_results_melt.head()


# In[47]:


cell_results_melt = cell_results_melt[cell_results_melt["cell_chip_overlap"] == "ChIP_overlap"]
cell_results_melt["type"] = "cell-type ChIP"
cell_results_melt = cell_results_melt[["type", "cell_type", "percent"]]

all_results_melt = all_results_melt[all_results_melt["all_chip_overlap"] == "ChIP_overlap"]
all_results_melt["type"] = "all ChIP"
all_results_melt = all_results_melt[["type", "cell_type", "percent"]]

results_melt = cell_results_melt.append(all_results_melt)
results_melt


# In[48]:


figs_dir = "figs/0__overlap_bar"
get_ipython().system('mkdir -p $figs_dir')

plt.figure(figsize=(1.8, 1.9))
sns.barplot(data=results_melt[results_melt["type"] == "all ChIP"], 
            x="cell_type", y="percent", color=sns.color_palette("Set2")[0])
plt.xlabel("")
plt.ylabel("% of motifs that overlap a ChIP peak")
plt.ylim((0, 100))
plt.savefig("%s/all_chip_overlap.fimo.pdf" % figs_dir, dpi="figure", bbox_inches="tight")


# In[ ]:




