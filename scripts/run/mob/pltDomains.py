import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cadaST.utils import mclust_R

sc.set_figure_params(dpi_save=150, vector_friendly=True)
sns.set_theme(style="white", font="Helvetica")

adata_path = "../dataset/MOB/MOBstereo_HV.h5ad"
adata_all_path = "../dataset/MOB/MOBstereo.h5ad"
impute_path = "../output/MOB/0426/a0.8_t0.2_i3_scale/imputed_exp.feather"
label_path = "../output/MOB/0426/a0.8_t0.2_i3_scale/labels.feather"
graphst_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/Graphst/results/MOB/MOB_S1.csv"
stagate_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/STAGATE/results/MOB/MOB_S1.csv"
fig_path = "/home/qinxianhan/project/spatial/cadast_demo/figure/MOB/"


def main():
    adata = sc.read_h5ad(
        "/home/qinxianhan/project/spatial/cadast_demo/output/MOB/0918/output.h5ad"
    )
    graphst_res = pd.read_csv(graphst_path, index_col=0, header=0)
    stagate_res = pd.read_csv(stagate_path, index_col=0, header=0)
    try:
        cadast_res = pd.read_csv("cadast_domains.csv", index_col=0)
    except FileNotFoundError:
        cadast_res = {}
        for cluster in range(6, 11):
            mclust_R(adata, cluster)
            cadast_res[cluster] = adata.obs["mclust"]
        cadast_res = pd.DataFrame(cadast_res)
        cadast_res.to_csv("cadast_domains.csv")

    fig, axs = plt.subplots(5, 3, figsize=(12, 15))
    for i in range(5):
        for j in range(3):
            res = stagate_res if j == 0 else graphst_res if j == 1 else cadast_res
            ndomains = str(i + 6)
            adata.obs["domain"] = res[ndomains]
            adata.obs["domain"] = adata.obs["domain"].astype("category")
            sc.pl.embedding(
                adata,
                title="",
                basis="spatial",
                color="domain",
                ax=axs[i, j],
                show=False,
                frameon=False,
            )

    for j in range(3):
        axs[0, j].set_title(["STAGATE", "GraphST", "Cadast"][j], fontsize=20)
    for i in range(5):
        axs[i, 0].text(
            -0.1,
            0.5,
            f"{i + 6} Clusters",
            fontsize=20,
            ha="right",
            va="center",
            transform=axs[i, 0].transAxes,
            rotation=90,
        )
    fig.tight_layout()
    fig.savefig(f"{fig_path}mob_domains_r.pdf")


if __name__ == "__main__":
    main()
