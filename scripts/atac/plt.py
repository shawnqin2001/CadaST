import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns


def plt_cluster(adata, colors=None, figsize=(5, 20),switch_axis=False, **kwargs):
    if colors is None:
        colors = ["CadaST", "STAGATE", "GraphST", "SpatialGlue"]
    fig, axs = plt.subplots(len(colors), 1, figsize=figsize)
    handles, labels = None, None
    for i, (ax, color) in enumerate(zip(axs, colors)):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=color,
            ax=ax,
            show=False,
            legend_loc="lower center" if i == len(colors) - 1 else None,
            palette=sns.color_palette("muted"),
            title="",
            **kwargs,
        )
        # ax.set_title(color, fontsize=26, weight="medium")
        ax.set_ylabel(color, fontsize=26, weight="medium")
        ax.set_xlabel("")
    handles, labels = axs[len(colors) - 1].get_legend_handles_labels()
    axs[len(colors) - 1].legend([], [], frameon=False)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=10,
        fontsize=10,
        markerscale=1,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )
    fig.tight_layout()
    return fig


def plt_gene_expression(adata, genelist):
    fontsize = 40
    size = 80
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    col_labels = ["Raw", "Predicted", "Denoised"]
    cmap = "viridis"
    for i, gene in enumerate(genelist):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="atac-" + gene,
            size=size,
            ax=axs[i, 0],
            show=False,
            colorbar_loc=None,
            legend_loc=None,
            title="",
            cmap=cmap,
        )
        axs[i, 0].axis("off")
    for i, gene in enumerate(genelist):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene + "_label",
            size=size,
            ax=axs[i, 1],
            show=False,
            colorbar_loc=None,
            legend_loc=None,
            title="",
            cmap=cmap,
        )
        axs[i, 1].axis("off")
    for i, gene in enumerate(genelist):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,
            size=size,
            ax=axs[i, 2],
            show=False,
            colorbar_loc=None,
            title="",
            cmap=cmap,
        )
        axs[i, 2].axis("off")

    for i, label in enumerate(genelist):
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
    for i, label in enumerate(col_labels):
        axs[0, i].set_title(label, fontsize=fontsize, weight="medium")
    fig.tight_layout()
    return fig


def plt_gene_umap(adata, genes, figsize, switch_axis=False, size=50, fontsize=26, **kwargs):
    size = size
    fontsize = fontsize
    if switch_axis:
        figsize = (figsize[1], figsize[0])
        fig, axs = plt.subplots(2, len(genes), figsize=figsize)
        for i, gene in enumerate(genes):
            gene_name = gene.replace("atac-", "")
            sc.pl.embedding(
                adata,
                basis="spatial",
                color=gene,
                ax=axs[0, i],
                size=size,
                show=False,
                **kwargs,
                cmap="viridis",
                colorbar_loc=None,
                title="",
            )
            sc.pl.umap(
                adata, color=gene, ax=axs[1, i], show=False, **kwargs, cmap="viridis", colorbar_loc=None, title=""
            )
        for ax in axs.flat:
            ax.set_xlabel("")
            ax.set_ylabel("")
        for i, gene in enumerate(genes):
            gene_name = gene.replace("atac-", "")
            axs[0, i].set_title(gene_name, fontsize=26, weight="medium", fontstyle="italic")
        axs[0, 0].set_ylabel("Spatial", fontsize=26, weight="medium")
        axs[1, 0].set_ylabel("UMAP", fontsize=26, weight="medium")
        fig.tight_layout()
        return fig
    fig, axs = plt.subplots(len(genes), 2, figsize=figsize)
    for i, gene in enumerate(genes):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,
            ax=axs[i, 0],
            size=size,
            show=False,
            **kwargs,
            cmap="viridis",
            colorbar_loc=None,
            title="",
        )
        sc.pl.umap(adata, color=gene, ax=axs[i, 1], show=False, **kwargs, cmap="viridis", title="")
    axs[0, 0].set_title("Spatial", fontsize=26, weight="medium")
    axs[0, 1].set_title("UMAP", fontsize=26, weight="medium")
    for ax in axs.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    for i, label in enumerate(genes):
        gene_name = label.replace("atac-", "")
        axs[i, 0].text(
            -0.05,
            0.5,
            gene_name,
            fontsize=fontsize,
            fontstyle="italic",
            ha="right",
            va="center",
            weight="medium",
            transform=axs[i, 0].transAxes,
            rotation=90,
        )
    fig.tight_layout()
    return fig


def plt_gene_label_umap(adata, genes, figsize, **kwargs):
    fig, axs = plt.subplots(len(genes), 3, figsize=figsize)
    for i, gene in enumerate(genes):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,
            ax=axs[i, 0],
            size=50,
            show=False,
            **kwargs,
            cmap="viridis",
            colorbar_loc=None,
            title="",
            layer="labels",
        )
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,
            ax=axs[i, 1],
            size=50,
            show=False,
            **kwargs,
            cmap="viridis",
            colorbar_loc=None,
            title="",
        )
        sc.pl.umap(adata, color=gene, ax=axs[i, 2], show=False, **kwargs, cmap="viridis", colorbar_loc=None, title="")
    axs[0, 0].set_title("Spatial", fontsize=26, weight="medium")
    axs[0, 1].set_title("UMAP", fontsize=26, weight="medium")
    for ax in axs.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
    for i, gene in enumerate(genes):
        gene_name = gene.replace("atac-", "")
        axs[i, 0].set_ylabel(gene_name, fontsize=26, weight="medium")
    fig.tight_layout()
    return fig
