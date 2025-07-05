import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter


def init_plot_params():
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "medium"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    sns.set_theme(context="notebook", style="white", font="Arial", font_scale=1.2)


init_plot_params()
stagate_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/stagate/results/embryo/STAGATE.csv"
graphst_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/Graphst/results/embryo/time.csv"
cadast_path = "/home/qinxianhan/project/spatial/cadast_demo/scripts/run/embryoStereo/time_reps.csv"


def load_data(path):
    df = pd.read_csv(path, index_col=0, header=0)
    return df


def main():
    colors = ["#FF6B6B", "#255FA7", "#336469"]
    cellnums = [5913, 18408, 30124, 51365, 77369, 102519, 113350, 121767]
    methodColors = {
        "STAGATE": colors[2],
        "GraphST": colors[1],
        "CadaST": colors[0],
    }

    # Load data
    cadast = load_data(cadast_path)
    stagate = load_data(stagate_path)
    graphst = load_data(graphst_path)

    # Calculate mean and std
    cadast_mean = cadast.mean(axis=1)
    cadast_std = cadast.std(axis=1)
    stagate_mean = stagate.mean(axis=1)
    stagate_std = stagate.std(axis=1)
    graphst_mean = graphst.mean(axis=1)
    graphst_std = graphst.std(axis=1)

    # Set font and style
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 16,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 11,
        }
    )

    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.6})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    markersize = 8
    linewidth_normal = 2.0
    linewidth_cadast = 3.0

    # Plot with error bars
    # CadaST
    ax.errorbar(
        cellnums[: len(cadast_mean)],
        cadast_mean,
        yerr=cadast_std,
        fmt="-v",
        color=methodColors["CadaST"],
        linewidth=linewidth_cadast,
        markersize=markersize + 2,
        capsize=5,
        label="CadaST",
        zorder=10,
    )

    # STAGATE
    ax.errorbar(
        cellnums[: len(stagate_mean)],
        stagate_mean,
        yerr=stagate_std,
        fmt="-o",
        color=methodColors["STAGATE"],
        linewidth=linewidth_normal,
        markersize=markersize,
        capsize=5,
        label="STAGATE",
    )

    # GraphST
    ax.errorbar(
        cellnums[: len(graphst_mean)],
        graphst_mean,
        yerr=graphst_std,
        fmt="-D",
        color=methodColors["GraphST"],
        linewidth=linewidth_normal,
        markersize=markersize,
        capsize=5,
        label="GraphST",
    )

    # Add edge to markers for better visibility
    for line in ax.lines:
        line.set_markeredgecolor("white")
        line.set_markeredgewidth(1.0)

    # Format axes
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Labels
    fontsize = 20
    ax.set_xlabel("Number of cells", fontweight="medium", fontsize=20)
    ax.set_ylabel("Running time (seconds)", fontweight="medium", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)

    # Legend
    legend = ax.legend(
        loc="upper left",
        frameon=False,
        title="Methods",
        bbox_to_anchor=(0.02, 1.0),
        title_fontsize=12,
        fontsize=11,
    )

    # Set y-axis limit
    all_means = [cadast_mean.min(), stagate_mean.min(), graphst_mean.min()]
    ax.set_ylim(bottom=min(all_means) * 0.8)

    sns.despine()
    fig.tight_layout()

    figpath = "/home/qinxianhan/project/spatial/cadast_demo/figures/sync/fig5/"
    plt.savefig(figpath + "time.pdf", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
