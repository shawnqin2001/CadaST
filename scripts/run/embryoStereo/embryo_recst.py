import os
import re
import pandas as pd
from anndata import read_h5ad
from cadaST.utils import mclust_R
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler

sc.set_figure_params(dpi=96, dpi_save=96, frameon=False, vector_friendly=True)
sns.set_theme(style="white", font="Helvetica")

today = "0312"
e13_path = "/home/qinxianhan/project/spatial/cadast_demo/output/embryo/1127_rep1/E13.5/adata.h5ad"
e14_path = (
    "/home/qinxianhan/project/spatial/cadast_demo/output/embryo/1127/adataE14.h5ad"
)
out_path = "../../output/embryo/{}/".format(today)

adataE13 = read_h5ad(e13_path)
adataE14 = read_h5ad(e14_path)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0


def compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def reclust() -> None:
    clustE13 = {}
    clustE14 = {}
    for clust in range(15, 25):
        print("clustering {}".format(clust))
        mclust_R(adataE13, clust)
        mclust_R(adataE14, clust)
        clustE13[clust] = adataE13.obs["mclust"]
        clustE14[clust] = adataE14.obs["mclust"]
        print("Clustering done for {}".format(clust))
    clustE13_df = pd.DataFrame(clustE13)
    clustE14_df = pd.DataFrame(clustE14)
    clustE13_df.to_csv(out_path + "clustE13.csv")
    clustE14_df.to_csv(out_path + "clustE14.csv")


def my_plot(adata: sc.AnnData, clust_df: pd.DataFrame, outpath: str) -> None:
    os.makedirs(outpath, exist_ok=True)
    clust_num = clust_df.shape[1]
    fig, axs = plt.subplots(2, int(clust_num / 2), figsize=(clust_num * 3, 12))
    axs = axs.flatten()
    for i in range(clust_num):
        clust = clust_df.columns[i]
        ax = axs[i]
        adata.obs["cadaST"] = clust_df[clust]
        adata.obs["cadaST"] = adata.obs["cadaST"].astype("category")
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="cadaST",
            ax=ax,
            show=False,
            size=20,
            frameon=False,
        )
        ax.invert_yaxis()
        ax.set_title("Cluster {}".format(clust), fontsize=30)
    fig.tight_layout()
    fig.savefig(outpath + "/clust.pdf")


def get_nmi(adata, clust_df, outpath):
    from sklearn.metrics import normalized_mutual_info_score

    nmi = {}
    for clust in clust_df.columns:
        nmi[clust] = normalized_mutual_info_score(
            adata.obs["annotation"], clust_df[clust], average_method="arithmetic"
        )
    nmi_df = pd.DataFrame({"NMI": nmi})
    nmi_df.to_csv(outpath + "/nmi.csv")


def get_chao(adata, clust_df, outpath):
    chaos = {}
    for clust in clust_df.columns:
        chaos[clust] = compute_CHAOS(clust_df[clust], adata.obsm["spatial"])
    chaos_df = pd.DataFrame({"CHAOS": chaos})
    chaos_df.to_csv(outpath + "/chaos.csv")


def my_barplot() -> None:
    """
    Create barplots for NMI and CHAOS metrics across different clustering solutions
    for both E13 and E14 datasets.
    """
    # Create output directory for plots
    plot_dir = out_path + "plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Read the metric files
    e13_nmi = pd.read_csv(out_path + "E13/nmi.csv", index_col=0)
    e13_chaos = pd.read_csv(out_path + "E13/chaos.csv", index_col=0)
    e14_nmi = pd.read_csv(out_path + "E14/nmi.csv", index_col=0)
    e14_chaos = pd.read_csv(out_path + "E14/chaos.csv", index_col=0)

    # Convert index to integers for cluster numbers
    e13_nmi.index = e13_nmi.index.astype(int)
    e13_chaos.index = e13_chaos.index.astype(int)
    e14_nmi.index = e14_nmi.index.astype(int)
    e14_chaos.index = e14_chaos.index.astype(int)

    # Create a figure with 2 subplots (NMI and CHAOS)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # Plot NMI values
    ax = axs[0]
    width = 0.35
    x = np.arange(len(e13_nmi))
    ax.bar(x - width / 2, e13_nmi["NMI"], width, label="E13.5")
    ax.bar(x + width / 2, e14_nmi["NMI"], width, label="E14.5")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("NMI score")
    ax.set_title("Normalized Mutual Information (NMI) for different cluster numbers")
    ax.set_xticks(x)
    ax.set_xticklabels(e13_nmi.index)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot CHAOS values
    ax = axs[1]
    ax.bar(x - width / 2, e13_chaos["CHAOS"], width, label="E13.5")
    ax.bar(x + width / 2, e14_chaos["CHAOS"], width, label="E14.5")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("CHAOS score")
    ax.set_title("CHAOS metric for different cluster numbers")
    ax.set_xticks(x)
    ax.set_xticklabels(e13_chaos.index)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Lower CHAOS values indicate better spatial coherence
    ax.text(
        0.02,
        0.98,
        "Note: Lower CHAOS values indicate better spatial coherence",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(plot_dir + "/metrics_comparison.pdf")
    plt.savefig(plot_dir + "/metrics_comparison.png", dpi=300)
    plt.close()

    print(f"Barplots saved to {plot_dir}/metrics_comparison.pdf and .png")


def main():
    # Make sure output directory exists
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_path + "E13", exist_ok=True)
    os.makedirs(out_path + "E14", exist_ok=True)

    # Check for clustering results and run if needed
    files = os.listdir(out_path)
    csv_files = [f for f in files if re.search(r"\.csv$", f)]
    if not csv_files:
        reclust()

    # Load clustering results
    clustE13_df = pd.read_csv(out_path + "clustE13.csv", index_col=0, header=0)
    clustE14_df = pd.read_csv(out_path + "clustE14.csv", index_col=0, header=0)

    # Generate spatial plots if they don't exist
    if not os.path.exists(out_path + "E13/clust.pdf"):
        print("Generating E13 spatial plots...")
        my_plot(adataE13, clustE13_df, out_path + "E13")

    if not os.path.exists(out_path + "E14/clust.pdf"):
        print("Generating E14 spatial plots...")
        my_plot(adataE14, clustE14_df, out_path + "E14")

    # Calculate NMI scores if they don't exist
    if not os.path.exists(out_path + "E13/nmi.csv"):
        print("Calculating E13 NMI scores...")
        get_nmi(adataE13, clustE13_df, out_path + "E13")

    if not os.path.exists(out_path + "E14/nmi.csv"):
        print("Calculating E14 NMI scores...")
        get_nmi(adataE14, clustE14_df, out_path + "E14")

    # Calculate CHAOS metric if it doesn't exist
    if not os.path.exists(out_path + "E13/chaos.csv"):
        print("Calculating E13 CHAOS metrics...")
        get_chao(adataE13, clustE13_df, out_path + "E13")

    if not os.path.exists(out_path + "E14/chaos.csv"):
        print("Calculating E14 CHAOS metrics...")
        get_chao(adataE14, clustE14_df, out_path + "E14")

    # Create barplots if all required metrics exist
    update_force = False
    if all(
        os.path.exists(f)
        for f in [
            out_path + "E13/nmi.csv",
            out_path + "E13/chaos.csv",
            out_path + "E14/nmi.csv",
            out_path + "E14/chaos.csv",
        ]
    ):
        plot_path = out_path + "plots/metrics_comparison.pdf"
        if not os.path.exists(plot_path) or update_force:
            print("Generating comparison barplots...")
            my_barplot()
        else:
            print(f"Barplots already exist at {plot_path}")
    else:
        print("Cannot generate barplots: some metric files are missing")


if __name__ == "__main__":
    main()
