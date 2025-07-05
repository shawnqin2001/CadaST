#  %%
import os
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams

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
fig_path = os.path.join(work_path, "figure/ATAC/e13/")


def plt_spatial(adata, color=None, size=80, **kwargs):
    if color is None and "domain" in adata.obs:
        color = "domain"
    sc.pl.embedding(adata, basis="spatial", color=color, size=size, frameon=False, **kwargs, cmap="viridis")


def plot_gene_expression(adata, genelist, size=80, figsize=(12, 8), transpose=False):
    fontsize = 28
    size = size
    labels = ["Predicted", "Denoised"]
    cmap = "viridis"
    if transpose:
        figsize = (figsize[1], figsize[0])
        fig, axs = plt.subplots(len(genelist), len(labels), figsize=figsize)
        for i, gene in enumerate(genelist):
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=gene + "_label",
                size=size,
                ax=axs[i, 0],
                show=False,
                colorbar_loc=None,
                legend_loc=None,
                title="",
                # frameon=False,
                cmap=cmap,
            )
            axs[i, 0].set_ylabel(ylabel=gene, fontsize=fontsize, weight="medium", fontstyle="italic", rotation=90)
            axs[i, 0].set_xlabel("")
        for i, gene in enumerate(genelist):
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=gene,
                size=size,
                ax=axs[i, 1],
                show=False,
                colorbar_loc=None,
                title="",
                # frameon=False,
                cmap=cmap,
            )

            axs[i, 1].set_ylabel("")
            axs[i, 1].set_xlabel("")
        for i, label in enumerate(labels):
            axs[0, i].set_title(label, fontsize=fontsize, weight="medium")
    else:
        fig, axs = plt.subplots(len(labels), len(genelist), figsize=figsize)
        for i, gene in enumerate(genelist):
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=gene + "_label",
                size=size,
                ax=axs[0, i],
                show=False,
                colorbar_loc=None,
                legend_loc=None,
                title="",
                frameon=False,
                cmap=cmap,
            )
            axs[0, i].set_title(label=gene, fontsize=fontsize, weight="medium", fontstyle="italic")
        for i, gene in enumerate(genelist):
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=gene,
                size=size,
                ax=axs[1, i],
                show=False,
                colorbar_loc=None,
                title="",
                frameon=False,
                cmap=cmap,
            )
        for i, label in enumerate(labels):
            axs[i, 0].text(
                -0.1,
                0.5,
                label,
                fontsize=fontsize,
                ha="right",
                va="center",
                weight="medium",
                transform=axs[i, 0].transAxes,
                rotation=90,
            )
    fig.tight_layout()
    return fig


# %%
adata = sc.read_h5ad(os.path.join(output_path, "embryoE13_feature.h5ad"))
print(adata)
# %%
genes = ["Dmrt3", "Cadm1", "Gad1"]

# %%
fig = plot_gene_expression(adata, genes, size=120, figsize=(12, 8), transpose=False)
fig.savefig(os.path.join(fig_path, "embryoE13_feature_genes_t.pdf"), bbox_inches="tight", dpi=80)
plt.close()
# %%
