# %%
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import colors
from matplotlib import rcParams

rcParams["font.family"] = "Arial"
rcParams["font.weight"] = "medium"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["savefig.bbox"] = "tight"
sns.set_theme(context="notebook", style="white", font="Arial", palette="muted")
sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)
adata_path = "/home/qinxianhan/project/spatial/cadast_demo/output/ATAC/embryoE13_processed.h5ad"
adata_path = "/home/qinxianhan/project/spatial/cadast_demo/output/ATAC/embryoE13_atac.h5ad"
gst_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/Graphst/results/atac"
stg_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/stagate/results/atac"
spg_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/spatialGlue/output/"
fig_path = "/home/qinxianhan/project/spatial/cadast_demo/figure/ATAC/"
# %%


def darken_color(color, amount=0.7):
    """
    Darkens the given color by multiplying (1 - amount) with each RGB component.
    amount=0.0 returns the original color
    amount=1.0 returns black
    """
    try:
        c = colors.to_rgb(color)
    except ValueError:
        raise ValueError(f"Invalid color: {color}")
    return tuple(max(c_i * amount, 0) for c_i in c)


set3_palette = sns.color_palette("Set3", n_colors=12)

palette = [darken_color(color, amount=0.9) for color in set3_palette]
# palette = sns.color_palette("muted", n_colors=12)
palette = sns.color_palette("muted")
palette[3], palette[7] = palette[7], palette[3]
palette[6], palette[9] = palette[9], palette[6]


def plt_cluster(adata, colors=None, **kwargs):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharex=True)
    handles, labels = None, None
    if colors is None:
        colors = ["CadaST", "STAGATE", "GraphST", "SpatialGlue"]
    for i, (ax, color) in enumerate(zip(axs, colors)):
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=color,
            ax=ax,
            show=False,
            legend_loc="right margin" if i == len(colors) - 1 else None,
            palette=palette,
            **kwargs,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(color, fontsize=26, weight="medium")
    handles, labels = axs[3].get_legend_handles_labels()
    axs[3].legend([], [], frameon=False)
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncol=1,
        fontsize=20,
        markerscale=1.5,
        bbox_to_anchor=(1.04, 0.5),
        frameon=False,
        # handlelength=3,
    )
    fig.tight_layout()
    return fig


# %%
adata = sc.read_h5ad(adata_path)
gst_domain = pd.read_csv(f"{gst_path}/e13_domain.csv", header=0)
stg_domain = pd.read_csv(f"{stg_path}/e13_domain.csv", header=0)
spg_domain = pd.read_csv(f"{spg_path}/e13_domain.csv", header=0)
adata.obs["GraphST"] = gst_domain.values
adata.obs["GraphST"] = adata.obs["GraphST"].astype("category")
adata.obs["STAGATE"] = stg_domain.values
adata.obs["STAGATE"] = adata.obs["STAGATE"].astype("category")
adata.obs["SpatialGlue"] = spg_domain.values
adata.obs["SpatialGlue"] = adata.obs["SpatialGlue"].astype("category")
adata.obs["CadaST"] = adata.obs["domain"].astype("category")

# %%
methods = ["CadaST", "STAGATE", "GraphST", "SpatialGlue"]
fig = plt_cluster(adata, colors=methods, size=180, alpha=0.95)
fig.savefig(f"{fig_path}/e13/clusters.pdf", bbox_inches="tight")
plt.close()
# %%
