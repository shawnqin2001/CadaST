#  %%
import os
import cadaST
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import numpy as np
from matplotlib import rcParams


# %%
rcParams["font.family"] = "Arial"
rcParams["font.weight"] = "medium"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["figure.figsize"] = (8, 8)
rcParams["savefig.bbox"] = "tight"
sns.set_theme(context="notebook", style="white", font="Arial", palette="muted")
sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)


def plt_spatial(adata, color=None, size=80, **kwargs):
    if color is None and "domain" in adata.obs:
        color = "domain"
    sc.pl.embedding(adata, basis="spatial", color=color, size=size, frameon=False, **kwargs, cmap="viridis")


# %%
work_path = "/home/qinxianhan/project/spatial/cadast_demo/"
output_path = os.path.join(work_path, "output/ATAC/")
data_path = os.path.join(work_path, "dataset/ATAC/")
adata = sc.read_h5ad(os.path.join(output_path, "embryoE13_processed.h5ad"))
adata_raw = sc.read_h5ad(os.path.join(data_path, "e13.h5ad"))
print(adata)
print(adata_raw)
# %%
gene = "atac-Gad1"
plt_spatial(adata, color=gene, layer="labels")
plt_spatial(adata, color=gene)
# %%
sc.pp.normalize_total(adata_raw, target_sum=1e5)
sc.pp.log1p(adata_raw)
sc.pp.scale(adata_raw, max_value=10)
# %%
adata_sub = adata_raw[:, adata.var_names].copy()
beta, alpha, theta, init_alpha = 300, 0.8, 0.1, 0
icm_iter = 3
max_iter = 5
n_components = 3
kneighbors = 4
sg = cadaST.SimilarityGraph(
    adata_sub,
    kneighbors=kneighbors,
    beta=beta,
    icm_iter=icm_iter,
    alpha=alpha,
    theta=theta,
    init_alpha=init_alpha,
    n_components=n_components,
    max_iter=max_iter,
)
# %%
gene = "atac-Cadm1"
sg.fit(gene)
adata_sub.obs["label"] = sg.labels
adata_sub.obs["exp"] = sg.exp
plt_spatial(adata_sub, color=[gene, "label", "exp"])

# %%
adata_sub.obs["Dmrt3"] = sg.exp.copy()
adata_sub.obs["Dmrt3_label"] = sg.labels.copy()
# %%
adata_sub.obs["Cadm1"] = sg.exp.copy()
adata_sub.obs["Cadm1_label"] = sg.labels.copy()
# %%
adata_sub.obs["Gad1"] = np.array(adata[:, "atac-Gad1"].X).reshape(-1)
adata_sub.obs["Gad1_label"] = np.array(adata[:, "atac-Gad1"].layers["labels"].reshape(-1))

# %%
adata_sub
# %%
genes = ["Dmrt3", "Cadm1", "Gad1"]
plt_spatial(adata_sub, color=genes, size=80)
plt_spatial(adata_sub, color=[gene + "_label" for gene in genes], size=80)
# %%
adata_sub.write_h5ad(os.path.join(output_path, "embryoE13_feature.h5ad"))
# %%
