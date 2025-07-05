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

# %%
import os
import cadaST
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from cadaST.utils import clustering
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


# %%
adata = sc.read_h5ad("/home/qinxianhan/project/spatial/cadast_demo/dataset/ATAC/emboryE13/emboryE13.h5ad")
adata_atac = adata[:, adata.var_names.str.contains("atac-")].copy()
adata_rna = adata[:, ~adata.var_names.str.contains("atac-")]
print(adata_atac.shape)

sc.pp.normalize_total(adata_atac, target_sum=1e5)
sc.pp.log1p(adata_atac)
sc.pp.scale(adata_atac, max_value=10)

sc.pp.pca(adata_atac)
sc.pp.neighbors(adata_atac)
sc.tl.umap(adata_atac)
sc.tl.leiden(adata_atac, resolution=0.4)
# %%
sc.pl.umap(adata_atac, color=["leiden"], frameon=False)
sc.pl.embedding(adata_atac, basis="spatial", frameon=False, color=["leiden"], size=100)
# %%
beta, alpha, theta, init_alpha = 1000, 0.8, 0.2, 1
icm_iter = 3
max_iter = 2
n_components = 3
kneighbors = 6
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
# %%
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
sc.pl.embedding(
    adata_atac, basis="spatial", color=["atac-Sox2"], frameon=False, size=150, show=False, ax=axs[0], cmap="viridis"
)
sc.pl.embedding(
    adata_fit, basis="spatial", color=["atac-Sox2"], frameon=False, size=150, show=False, ax=axs[1], cmap="viridis"
)
sc.pl.embedding(
    adata_fit,
    basis="spatial",
    color=["atac-Sox2"],
    frameon=False,
    size=150,
    layer="labels",
    show=False,
    ax=axs[2],
    cmap="viridis",
)
fig.show()
# %%
sc.pp.pca(adata_fit, n_comps=16)
sc.pp.neighbors(adata_fit)
sc.tl.umap(adata_fit)
sc.tl.leiden(adata_fit, resolution=0.2)
sc.pl.umap(adata_fit, color=["leiden"], frameon=False)
sc.pl.embedding(adata_fit, basis="spatial", frameon=False, color=["leiden"], size=100)

# %%
clustering(adata_fit, n_clusters=10)
sc.pl.embedding(adata_fit, basis="spatial", frameon=False, color=["domain"], size=100)
# %%

sc.pp.neighbors(adata_fit, use_rep="X_pca")
sc.tl.umap(adata_fit)
sc.pl.umap(adata_fit, color=["domain"], frameon=False)

# %%
out_path = "/home/qinxianhan/project/spatial/cadast_demo/output/ATAC"
adata_fit.write(os.path.join(out_path, "emboryE13_processed.h5ad"))
# %%
sc.pl.embedding(
    adata_fit,
    basis="spatial",
    frameon=False,
    color=["domain", "RNA_clusters", "ATAC_clusters", "Joint_clusters"],
    size=100,
)

# %%
