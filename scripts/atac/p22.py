# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: cadast
#     language: python
#     name: python3
# ---

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cadaST

sns.set_theme(style="whitegrid", palette="muted")
data_path = "/home/qinxianhan/project/spatial/cadast_demo/dataset/ATAC/emboryP22/"

adata = sc.read_h5ad(data_path + "emboryP22.h5ad")
plt.rcParams["figure.figsize"] = (5, 5)
sc.pl.embedding(adata, basis="spatial", frameon=False, color=["ATAC_clusters"], size=30)

adata_atac = adata[:, adata.var_names.str.contains("atac-")].copy()
print(f"Number of ATAC features: {adata_atac.shape[1]}")
sc.pp.normalize_total(adata_atac, target_sum=1e5)
sc.pp.log1p(adata_atac)
sc.pp.scale(adata_atac, max_value=10)

# sc.pp.highly_variable_genes(adata_atac, flavor="seurat_v3", n_top_genes=5000)
sc.pp.pca(adata_atac)
sc.pp.neighbors(adata_atac)
sc.tl.umap(adata_atac)

sc.tl.leiden(adata_atac, resolution=0.1, flavor="igraph", n_iterations=2)
sc.pl.umap(adata_atac, color=["leiden"], frameon=False)

sc.pl.embedding(adata_atac, basis="spatial", frameon=False, color=["leiden"], size=30)

beta, alpha, theta, init_alpha = 1000, 0.6, 0.2, 1
icm_iter = 3
max_iter = 3
n_components = 3
kneighbors = 16
sg = cadaST.SimilarityGraph(
    adata_atac,
    kneighbors=kneighbors,
    beta=beta,
    alpha=alpha,
    theta=theta,
    init_alpha=init_alpha,
    n_components=n_components,
)

target_gene = "atac-Sox4"
sg.fit(target_gene)
adata_atac.obs["labels"] = sg.labels
adata_atac.obs["imputed"] = sg.exp
sc.pl.embedding(
    adata_atac, basis="spatial", frameon=False, color=[target_gene, "labels", "imputed"], size=30, cmap="viridis"
)

beta, alpha, theta, init_alpha = 1000, 0.6, 0.2, 1
icm_iter = 2
max_iter = 4
n_components = 3
kneighbors = 16
n_top = 2000
model = cadaST.CadaST(
    adata_atac,
    kneighbors=kneighbors,
    beta=beta,
    icm_iter=icm_iter,
    alpha=alpha,
    theta=theta,
    init_alpha=init_alpha,
    n_components=n_components,
    max_iter=max_iter,
    n_top=n_top,
)
adata_fit = model.fit()

from cadaST.utils import clustering

clustering(adata_fit, n_clusters=10)
sc.pl.embedding(adata_fit, basis="spatial", frameon=False, color=["domain"], size=30)

sc.pp.pca(adata_fit)
sc.pp.neighbors(adata_fit)
sc.tl.umap(adata_fit)
sc.tl.leiden(adata_fit, resolution=0.2, flavor="igraph", n_iterations=2)
sc.pl.umap(adata_fit, color=["leiden"], frameon=False)
sc.pl.embedding(adata_fit, basis="spatial", frameon=False, color=["leiden"], size=30)

# Save the processed ATAC data
output_path = data_path + "emboryP22_processed.h5ad"
adata_fit.write(output_path)
