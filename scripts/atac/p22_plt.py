#  %%
import os
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams
from .plt import plt_cluster, plt_gene_label_umap, plt_gene_umap

rcParams["font.family"] = "Arial"
rcParams["font.weight"] = "medium"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["figure.figsize"] = (8, 8)
rcParams["savefig.bbox"] = "tight"
sns.set_theme(context="notebook", style="white", font="Arial", palette="muted")
sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)
work_path = "/home/qinxianhan/project/spatial/cadast_demo/"
output_path = os.path.join(work_path, "output/ATAC/")
data_path = os.path.join(work_path, "dataset/ATAC/")
fig_path = os.path.join(work_path, "figure/ATAC/embryoP22/")
gst_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/Graphst/results/atac"
stg_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/stagate/results/atac"
spg_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/spatialGlue/output/"
fig_path = "/home/qinxianhan/project/spatial/cadast_demo/figures/ATAC/p22/"


# %%
adata = sc.read_h5ad(os.path.join(output_path, "embryoP22_processed.h5ad"))
adata_raw = sc.read_h5ad("/home/qinxianhan/project/spatial/cadast_demo/dataset/ATAC/p22.h5ad")
print(adata)
# %%
gst_domain = pd.read_csv(f"{gst_path}/p22_domain.csv", header=0)
stg_domain = pd.read_csv(f"{stg_path}/p22_domain.csv", header=0)
spg_domain = pd.read_csv(f"{spg_path}/p22_domain.csv", header=0)
adata.obs["GraphST"] = gst_domain.values
adata.obs["GraphST"] = adata.obs["GraphST"].astype("category")
adata.obs["STAGATE"] = stg_domain.values
adata.obs["STAGATE"] = adata.obs["STAGATE"].astype("category")
adata.obs["SpatialGlue"] = spg_domain.values
adata.obs["SpatialGlue"] = adata.obs["SpatialGlue"].astype("category")
adata.obs["CadaST"] = adata.obs["leiden"].astype(int) + 1
adata.obs["CadaST"] = adata.obs["CadaST"].astype("category")

# %%
methods = ["STAGATE", "GraphST", "SpatialGlue"]
fig = plt_cluster(adata, colors=methods, figsize=(5, 5 * len(methods)), size=50, alpha=0.9)
fig.savefig(f"{fig_path}/clusters_3.pdf", bbox_inches="tight", dpi=80)
# plt.close()

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sc.pl.embedding(
    adata,
    basis="spatial",
    color="CadaST",
    show=False,
    ax=axs[0],
    legend_loc=None,
    size=40,
    alpha=0.9,
    title="",
)

sc.pl.umap(adata, color="CadaST", show=False, ax=axs[1], legend_loc="right margin", title="UMAP")
axs[0].set_title("Spatial Clusters", fontsize=20, weight="medium")
axs[1].set_title("UMAP", fontsize=20, weight="medium")
# handles, labels = axs[1].get_legend_handles_labels()
# axs[1].legend(
#     handles=handles,
#     labels=labels,
#     loc="center right",
#     # bbox_to_anchor=(0.5, -0.2),
#     ncol=1,
#     fontsize=12,
#     frameon=False,
# )
for ax in axs:
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.tight_layout()
fig.savefig(f"{fig_path}/cadast_cluster_umap_r.pdf", bbox_inches="tight", dpi=150)
# plt.close()


# %%
genes = ["Sept7", "Isl1", "Dlx1"]
genes = ["atac-" + gene for gene in genes]
fig = plt_gene_umap(adata, genes, figsize=(11, 15), switch_axis=True)
fig.savefig(f"{fig_path}/genes_umap_t.pdf", bbox_inches="tight", dpi=150)
# plt.close()

# %%
genes = ["Sept7", "Isl1", "Dlx1"]
genes = ["atac-" + gene for gene in genes]
fig = plt_gene_label_umap(adata, genes, figsize=(15, 15))
fig.savefig(f"{fig_path}/genes_labels_umap.pdf", bbox_inches="tight", dpi=80)
plt.close()
