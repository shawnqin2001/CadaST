import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams


def init_plot_params():
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "medium"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    sns.set_theme(context="notebook", style="white", font="Arial", font_scale=1.2)


init_plot_params()

stagate_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/stagate/results/embryo/STAGATE.csv"
graphst_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/Graphst/results/embryo/time.csv"
cadast_path = "/home/qinxianhan/project/spatial/cadast_demo/script/run/embryoStereo/time_reps.csv"


def load_data(path):
    files = os.listdir(path)
    files_cpu = [x for x in files if "gpu" not in x]
    # files_gpu = [x for x in files if "gpu" in x]
    files_cpu.sort()
    # files_gpu.sort()
    times_cpu = {}
    # times_gpu = {}
    for file in files_cpu:
        with open(os.path.join(path, file), "r") as f:
            time = f.readlines()[0]
            times_cpu[file.split("_")[0]] = float(time)
    # for file in files_gpu:
    #     with open(os.path.join(path, file), "r") as f:
    #         time = f.readlines()[0]
    #         times_gpu[file.split("_")[0]] = float(time)
    return pd.Series(times_cpu)


def main():
    colors = ["#FF6B6B", "#255FA7", "#336469"]

    methodColors = {
        "STAGATE": colors[2],
        "GraphST": colors[1],
        "CadaST": colors[0],
    }
    stagate_cpu = load_data(stagate_path)
    graphst_cpu = load_data(graphst_path)
    cadast_cpu = pd.read_csv(cadast_path, index_col=0).squeeze("columns")

    df_time = pd.concat(
        [
            stagate_cpu.rename("STAGATE_cpu"),
            graphst_cpu.rename("GraphST_cpu"),
            cadast_cpu.rename("CadaST_cpu"),
        ],
        axis=1,
    )
    df_time = df_time.sort_index()
    print(df_time)
    cellnums = [5913, 18408, 30124, 51365, 77369, 102519, 113350, 121767]
    df_time["cellnum"] = cellnums

    # 设置字体和风格
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 11,
        }
    )

    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.6})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    markers = ["o", "s", "D", "^", "v"]
    markersize = 8

    # 定义线条样式：CPU为实线，GPU为虚线
    linewidth_normal = 2.0
    linewidth_cadast = 3.0  # CadaST使用更粗的线条

    # 绘制线条，按照CPU实线/GPU虚线的要求，相同方法用相同颜色
    # STAGATE CPU
    sns.lineplot(
        data=df_time,
        x="cellnum",
        y="STAGATE_cpu",
        ax=ax,
        label="STAGATE",
        markersize=markersize,
        marker="o",
        color=methodColors["STAGATE"],
        linewidth=linewidth_normal,
        linestyle="-",  # CPU实线
        alpha=0.9,
    )

    # STAGATE GPU - 与CPU相同颜色
    # sns.lineplot(
    #     data=df_time,
    #     x="cellnum",
    #     y="STAGATE_gpu",
    #     ax=ax,
    #     label="STAGATE (GPU)",
    #     markersize=markersize,
    #     marker="s",
    #     color=methodColors["STAGATE"],
    #     linewidth=linewidth_normal,
    #     linestyle="--",  # GPU虚线
    #     alpha=0.9,
    # )

    # GraphST CPU
    sns.lineplot(
        data=df_time,
        x="cellnum",
        y="GraphST_cpu",
        ax=ax,
        label="GraphST",
        markersize=markersize,
        marker="D",
        color=methodColors["GraphST"],
        linewidth=linewidth_normal,
        linestyle="-",  # CPU实线
        alpha=0.9,
    )

    # GraphST GPU - 与CPU相同颜色
    # sns.lineplot(
    #     data=df_time,
    #     x="cellnum",
    #     y="GraphST_gpu",
    #     ax=ax,
    #     label="GraphST (GPU)",
    #     markersize=markersize,
    #     marker="^",
    #     color=methodColors["GraphST"],
    #     linewidth=linewidth_normal,
    #     linestyle="--",  # GPU虚线
    #     alpha=0.9,
    # )

    # CadaST (突出显示) - 单独一种颜色
    sns.lineplot(
        data=df_time,
        x="cellnum",
        y="CadaST_cpu",
        ax=ax,
        label="CadaST (CPU)",
        markersize=markersize + 2,  # 更大的标记
        marker="v",
        color=methodColors["CadaST"],
        linewidth=linewidth_cadast,  # 更粗的线条
        linestyle="-",  # CPU实线
        alpha=1.0,  # 完全不透明
        zorder=10,  # 确保CadaST在最上层
    )

    # 添加轻微的阴影以增强视觉效果
    for line in ax.lines:
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(1.0)

    # 使用对数刻度以便于显示大范围的数据
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # 美化标签
    fontsize = 20
    ax.set_xlabel("Number of cells", fontweight="medium", fontsize=20)
    ax.set_ylabel("Running time (seconds)", fontweight="medium", fontsize=20)
    # Set tick parameters to change font size
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    # ax.set_title(
    # "Performance Comparison Across Cell Numbers", fontweight="bold", pad=15)

    # 修改图例样式，将Method和Device分成两个部分
    from matplotlib.lines import Line2D

    # 移除默认图例
    ax.get_legend().remove()

    # 创建自定义图例项 - 分成两组
    method_elements = [
        # 三种方法的颜色标识
        Line2D([0], [0], color=methodColors["STAGATE"], lw=linewidth_normal, label="STAGATE"),
        Line2D([0], [0], color=methodColors["GraphST"], lw=linewidth_normal, label="GraphST"),
        Line2D([0], [0], color=methodColors["CadaST"], lw=linewidth_cadast, label="CadaST"),
    ]

    # device_elements = [
    #     # CPU/GPU的线型标识（使用黑色以突出线型区别）
    #     Line2D([0], [0], color="black", lw=linewidth_normal, linestyle="-", label="CPU"),
    #     Line2D([0], [0], color="black", lw=linewidth_normal, linestyle="--", label="GPU"),
    # ]

    # 创建两个分开的图例，确保标题对齐
    legend_title_fontsize = 12
    legend_fontsize = 11

    # 第一个图例
    method_legend = ax.legend(
        handles=method_elements,
        loc="upper left",
        frameon=False,
        title="Methods",
        bbox_to_anchor=(0.02, 1.0),
        title_fontsize=legend_title_fontsize,
        fontsize=legend_fontsize,
    )
    method_legend._legend_box.align = "left"

    # 添加第二个图例，确保与第一个标题对齐
    # ax.add_artist(method_legend)
    # device_legend = ax.legend(
    #     handles=device_elements,
    #     loc="upper left",
    #     frameon=False,
    #     title="Devices",
    #     bbox_to_anchor=(0.02, 0.80),  # 位置在第一个图例下方
    #     title_fontsize=legend_title_fontsize,
    #     fontsize=legend_fontsize,
    # )
    # device_legend._legend_box.align = "left"

    # 设置网格线
    # ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    # ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

    # 设置轴范围以确保数据更好地展示
    ax.set_ylim(bottom=df_time[["STAGATE_cpu", "GraphST_cpu", "CadaST_cpu"]].min().min() * 0.8)

    # 添加轻微的背景色
    # ax.set_facecolor("#f8f9fa")

    # 为CadaST添加一个特别的注释，使用CadaST颜色
    # max_cadast_y = df_time["CadaST_cpu"].max()
    # max_cadast_x = df_time.loc[df_time["CadaST_cpu"].idxmax(), "cellnum"]

    # ax.annotate(
    #     "CadaST shows superior\nperformance",
    #     xy=(max_cadast_x, max_cadast_y),
    #     xytext=(max_cadast_x * 0.7, max_cadast_y * 0.5),
    #     arrowprops=dict(
    #         facecolor=methodColors["CadaST"], shrink=0.05, width=2, alpha=0.8
    #     ),
    #     fontsize=10,
    #     fontweight="bold",
    #     color=methodColors["CadaST"],
    #     ha="center",
    # )

    sns.despine()
    fig.tight_layout()
    figpath = "/home/qinxianhan/project/spatial/cadast_demo/figure/sync/fig5/"
    plt.savefig(figpath + "time.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
