import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from params import init_plot_params

init_plot_params()
work_dir = "/home/qinxianhan/project/spatial/cadast_demo/"
adata_all_path = f"{work_dir}/dataset/MOB/MOBstereo.h5ad"
impute_path = f"{work_dir}/output/MOB/0426/a0.8_t0.2_i3_scale/imputed_exp.feather"
label_path = f"{work_dir}/output/MOB/0426/a0.8_t0.2_i3_scale/labels.feather"
fig_path = f"{work_dir}/figure/MOB/"
data_path = f"{work_dir}/output/MOB/0918/output.h5ad"
output_format = "both"  # Options: "png", "pdf", "both"


def save_figure(fig, filename, dpi=96):
    filename = os.path.join(fig_path, filename)
    print("Saving figure to", filename)
    if output_format in ["png", "both"]:
        fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")
    if output_format in ["pdf", "both"]:
        fig.savefig(f"{filename}.pdf", dpi=dpi, bbox_inches="tight")


def plot_cluster(adata):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="STAGATE",
        ax=axs[0],
        show=False,
        frameon=False,
        size=12,
        legend_loc=None,
    )
    axs[0].set_title("STAGATE", fontsize=26, weight="medium")
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="GraphST",
        ax=axs[1],
        show=False,
        frameon=False,
        size=12,
        legend_loc=None,
    )
    axs[1].set_title("GraphST", fontsize=26, weight="medium")
    sc.pl.embedding(
        adata,
        basis="spatial",
        color="domain",
        ax=axs[2],
        show=False,
        frameon=False,
        size=12,
    )
    axs[2].set_title("CadaST", fontsize=26, weight="medium")

    fig.tight_layout()
    save_figure(fig, "compare", dpi=150)
    plt.close()


def plot_domain(adata, domains):
    fig, axs = plt.subplots(3, 7, figsize=(35, 11))
    annotOrder = ["ONL", "GL", "EPL", "MCL", "IPL", "GCL", "RMS"]
    gstOrder = [6, 7, 1, 5, 4, 2, 3]
    stagateOrder = [3, 7, 2, 6, 4, 1, 5]
    spotSize = 20
    row_labels = ["CadaST", "GraphST", "STAGATE"]
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(7):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="annot",
            ax=axs[0][i],
            groups=[annotOrder[i]],
            show=False,
            size=spotSize,
            legend_loc=None,
            title="",
            frameon=False,
        )
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="GraphST",
            ax=axs[1][i],
            groups=[gstOrder[i]],
            show=False,
            size=spotSize,
            legend_loc=None,
            title="",
            frameon=False,
        )
        sc.pl.embedding(
            adata,
            basis="spatial",
            color="STAGATE",
            ax=axs[2][i],
            groups=[stagateOrder[i]],
            show=False,
            size=spotSize,
            legend_loc=None,
            title="",
            frameon=False,
        )
        axs[0][i].set_title(annotOrder[i], fontsize=35, fontweight="medium")

    for i, label in enumerate(row_labels):
        axs[i, 0].text(
            -0.05,
            0.5,
            label,
            fontsize=35,
            ha="right",
            va="center",
            transform=axs[i, 0].transAxes,
            rotation=90,
            fontweight="medium",
        )

    fig.tight_layout()
    save_figure(fig, "domains", dpi=96)


def plot_chaos():
    stcada = np.load(f"{work_dir}/output/MOB/CHAOS.npy")
    graphst = np.load(f"{work_dir}benchmark/Graphst/results/MOB/CHAOS.npy")
    stagate = np.load(f"{work_dir}benchmark/STAGATE/results/MOB/CHAOS.npy")

    # 创建DataFrame
    CHAOS_df = pd.DataFrame({"CadaST": stcada, "STAGATE": stagate, "GraphST": graphst})
    CHAOS_df.set_index(pd.Index(range(6, 11), name="Domains"), inplace=True)
    CHAOS_melt = CHAOS_df.melt(var_name="Method", value_name="CHAOS", ignore_index=False)
    CHAOS_melt.reset_index(inplace=True)
    sns.set_style("white")
    # 定义美化后的颜色方案
    cusPalette = ["#FF6B6B", "#336469", "#255FA7"]

    # cusPalette = sns.color_palette("deep", 3)
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制柱状图并美化
    bar_plot = sns.barplot(
        data=CHAOS_melt,
        x="Domains",
        y="CHAOS",
        hue="Method",
        palette=cusPalette,
        width=0.8,
        alpha=0.9,
        edgecolor="#333333",
        linewidth=3,
        errorbar=None,
    )

    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=10, fontweight="light", padding=5)

    # 美化坐标轴
    ax.set_xlabel("", fontsize=22, fontweight="medium")
    ax.set_ylabel("CHAOS Score", fontsize=22, fontweight="medium")

    # 调整y轴范围，留出空间给数值标签
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(0.015, current_ymax)

    # 美化刻度标签
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("medium")

    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 美化图例
    legend = ax.legend(
        title="",
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#333333",
        fontsize=16,
        title_fontsize=18,
    )
    legend.get_frame().set_linewidth(1.5)

    # 添加网格线
    # ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.grid(visible=False)
    # 添加标题
    # ax.set_title("CHAOS Score Comparison", fontsize=24, fontweight="medium", pad=20)

    # 紧凑布局
    fig.tight_layout()

    # 保存图形
    save_figure(fig, "CHAOS", dpi=300)


def main():
    adata = sc.read(data_path)
    staMeta = pd.read_csv(f"{work_dir}benchmark/STAGATE/results/MOB/STAGATE_louvain7.csv", index_col=0)
    gstMeta = pd.read_csv(f"{work_dir}benchmark/Graphst/results/MOB/GraphST.csv", index_col=0)
    adata.obs["STAGATE"] = staMeta["louvain"].astype("category")
    adata.obs["GraphST"] = gstMeta["domain"].astype("category")
    adata.obs["CadaST"] = adata.obs["domain"].astype("category")
    adata.obs["domain"] = adata.obs["domain"].astype(str)
    domains = ["GCL", "EPL", "IPL", "GL", "ONL", "MCL", "RMS"]
    annoMap = {str(i + 1): domain for i, domain in enumerate(domains)}
    # adata.obs["annot"] = adata.obs["domain"].map(annoMap)
    # print(adata)
    # plot_cluster(adata)
    # plot_domain(adata, domains)
    plot_chaos()


if __name__ == "__main__":
    main()
