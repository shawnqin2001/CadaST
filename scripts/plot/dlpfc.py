import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from cadaST import CadaST
from params import init_plot_params

init_plot_params()

sample = "DLPFC"
dataset_path = f"../../dataset/{sample}/"
figure_path = "/home/qinxianhan/project/spatial/cadast_demo/figure/DLPFC"
sampleid = "151673"


def main():
    adata_raw = sc.read_h5ad(f"{dataset_path}/{sampleid}_processed.h5ad")
    beta, alpha, theta = 12, 0.6, 0.2
    max_iter = 2
    icm_iter = 2
    kneighbors = 24
    model = CadaST(
        adata_raw,
        beta=beta,
        kneighbors=kneighbors,
        max_iter=max_iter,
        icm_iter=icm_iter,
        alpha=alpha,
        theta=theta,
        n_top=2000,
        n_jobs=24,
    )
    adata = model.fit()
    marker_plot(adata, adata_raw, figure_path)


def marker_plot(
    adata: AnnData,
    adata_raw: AnnData,
    save_path: str = None,
):
    marker_genes = ["PLP1", "PCP4", "CAMK2N1"]
    fig, axs = plt.subplots(3, 3)
    row_labels = ["Raw", "Predicted", "Denoised"]
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fontsize = 10
    cmap = "viridis"
    # sns.set_theme(color_codes='b')
    for i, gene in enumerate(marker_genes):
        sc.pl.spatial(
            adata_raw,
            color=gene,
            img_key=None,
            size=1.6,
            ax=axs[0, i],
            show=False,
            colorbar_loc=None,
            cmap=cmap,
            title="",
        )
        axs[0, i].axis("off")
        axs[0, i].set_title(gene, fontsize=fontsize)
    for i, gene in enumerate(marker_genes):
        sc.pl.spatial(
            adata,
            color=gene,
            img_key=None,
            size=1.6,
            ax=axs[1, i],
            show=False,
            colorbar_loc=None,
            layer="labels",
            title="",
            cmap=cmap,
        )
        axs[1, i].axis("off")
    for i, gene in enumerate(marker_genes):
        sc.pl.spatial(
            adata,
            color=gene,
            img_key=None,
            size=1.6,
            ax=axs[2, i],
            show=False,
            colorbar_loc=None,
            title="",
            cmap=cmap,
        )
        axs[2, i].axis("off")

    for i, label in enumerate(row_labels):
        axs[i, 0].text(
            -0.1,
            0.5,
            label,
            fontsize=fontsize,
            ha="right",
            va="center",
            transform=axs[i, 0].transAxes,
            rotation=90,
        )
    plt.tight_layout(pad=0.01)
    if save_path:
        print("Saving plot to {}".format(save_path))
        plt.savefig(f"{save_path}/markers.pdf", bbox_inches="tight", dpi=96)


if __name__ == "__main__":
    main()
