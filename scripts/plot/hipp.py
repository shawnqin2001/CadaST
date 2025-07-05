import os
import scanpy as sc
import matplotlib.pyplot as plt
from params import init_plot_params

init_plot_params()
work_dir = "/home/qinxianhan/project/spatial/cadast_demo/"
data_path = f"{work_dir}/output/slide/1126/out_filter.h5ad"
fig_path = f"{work_dir}/figure/Hipp/"

dpi = 96
output_format = "pdf"  # Options: "png", "pdf", "both"


def main():
    adata = sc.read(data_path)
    print(adata)
    # plot_clusters(adata)
    genelist = ["Wfs1", "Strip2", "Pvrl3", "C1ql2"]
    plot_gene_expression(adata, genelist)
    return


def save_figure(fig, filename):
    filename = os.path.join(fig_path, filename)
    print("Saving figure to", filename)
    if output_format in ["png", "both"]:
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    if output_format in ["pdf", "both"]:
        fig.savefig(f"{filename}.pdf", dpi=dpi, bbox_inches="tight")


def plot_gene_expression(adata, gene_list):
    fontsize = 40
    genelist = ["Wfs1", "Strip2", "Pvrl3", "C1ql2"]
    fig, axs = plt.subplots(3, 4, figsize=(12, 10))
    row_labels = ["Raw", "Predicted", "Denoised"]
    adata_raw = sc.read_h5ad("/home/qinxianhan/project/spatial/cadast_demo/dataset/slide_hipp/raw.h5ad")
    adata_raw = adata_raw[adata.obs_names,]
    cmap = "viridis"
    for i, gene in enumerate(genelist):
        sc.pl.embedding(
            adata_raw,
            basis="spatial",
            color=gene,
            size=5,
            ax=axs[0, i],
            show=False,
            colorbar_loc=None,
            legend_loc=None,
            cmap=cmap,
        )
        axs[0, i].axis("off")
        axs[0, i].invert_yaxis()
        axs[0, i].set_title(label=genelist[i], fontsize=fontsize, weight="medium")
    for i, gene in enumerate(genelist):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,
            size=4,
            ax=axs[1, i],
            show=False,
            colorbar_loc=None,
            legend_loc=None,
            layer="labels",
            title="",
            cmap=cmap,
        )
        axs[1, i].axis("off")
        axs[1, i].invert_yaxis()
    for i, gene in enumerate(genelist):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,
            size=5,
            ax=axs[2, i],
            show=False,
            colorbar_loc=None,
            title="",
            cmap=cmap,
        )
        axs[2, i].axis("off")
        axs[2, i].invert_yaxis()

    for i, label in enumerate(row_labels):
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
    save_figure(fig, "markers")
    plt.close()


def plot_clusters(adata):
    methods = ["STAGATE", "GraphST", "CadaST"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    size = 15
    palette = "tab20"
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="STAGATE",
        ax=axs[0],
        show=False,
        frameon=False,
        legend_loc=None,
        size=size,
        palette=palette,
    )
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="GraphST",
        ax=axs[1],
        show=False,
        frameon=False,
        legend_loc=None,
        size=size,
        palette=palette,
    )
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="CadaST",
        ax=axs[2],
        show=False,
        frameon=False,
        size=size,
        palette=palette,
    )

    for i, ax in enumerate(axs):
        ax.invert_yaxis()
        ax.set_title(methods[i], fontsize=26, weight="medium")
    handles, labels = axs[2].get_legend_handles_labels()
    axs[2].legend([], [], frameon=False)
    fig.legend(
        handles,
        labels,
        ncol=1,
        fontsize=13,
        markerscale=1,
        frameon=False,
        handletextpad=0.3,
        bbox_to_anchor=(0.98, 0.5),  # Position relative to figure (x, y)
        loc="center left",
    )
    fig.tight_layout()
    save_figure(fig, "clusters")
    plt.close()


if __name__ == "__main__":
    main()
