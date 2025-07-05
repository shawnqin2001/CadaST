import scanpy as sc
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from cadaST.utils import clustering

sc.set_figure_params(dpi_save=96, vector_friendly=True)


def my_clust(adata, num_clust):
    adata = adata.copy()
    timepoint = adata.obs["timepoint"].unique()[0]
    target_clust = min(num_clust[timepoint], 20)
    clustering(adata, target_clust)
    return adata


if __name__ == "__main__":
    try:
        adata_all = sc.read_h5ad("/home/qinxianhan/project/spatial/cadast_demo/pyscript/embryoStereo/adata_all.h5ad")
    except FileNotFoundError:
        print("adata_all not found, start to generate")
        adata_raw = sc.read_h5ad(
            "/home/qinxianhan/project/spatial/dataset/MouseEmbryo/Mouse_embryo_all_stage.h5ad",
            backed="r",
        )
        timepoints = set(adata_raw.obs["timepoint"])
        outPath = "/home/qinxianhan/project/spatial/cadast_demo/output/embryo/1125"
        adatas = [sc.read_h5ad(os.path.join(outPath, timepoint, "adata.h5ad")) for timepoint in timepoints]
        num_clust = {}
        for timepoint in adata_raw.obs["timepoint"].unique():
            adata = adata_raw[adata_raw.obs["timepoint"] == timepoint]
            num_clust[timepoint] = len(adata.obs["annotation"].unique())

        results = Parallel(n_jobs=4)(delayed(my_clust)(adata, num_clust) for adata in adatas)
        adata_all = sc.AnnData.concatenate(*results)
        print(adata_all)
        adata_all.write_h5ad("/home/qinxianhan/project/spatial/cadast_demo/pyscript/embryoStereo/adata_all.h5ad")
    print(adata_all)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    sc.pl.embedding(
        adata_all,
        basis="spatial",
        color="domain",
        show=False,
        size=25,
        palette="tab20",
        legend_loc=None,
        ax=ax,
    )
    ax.invert_yaxis()
    ax.axis("off")
    fig.savefig(
        "/home/qinxianhan/project/spatial/cadast_demo/pyscript/embryoStereo/embryo_stereo.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
