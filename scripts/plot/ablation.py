# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import warnings
import plottable as pt
import os
from matplotlib import rcParams
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, silhouette_score
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("ignore")
sns.set_theme(style="white", font="Arial")


# +


def init_plot_params():
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "medium"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["savefig.bbox"] = "tight"
    sns.set_theme(context="notebook", style="white", font="Arial", palette="muted")
    sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)


init_plot_params()
# -

samples = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]

DATAPATH = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/analysis/data/DLPFC"
adatas = []
for sample in samples:
    adata = sc.read_visium(f"{DATAPATH}/{sample}")
    adatas.append(adata)

# +
meta_path = "/home/qinxianhan/project/spatial/cadast_demo/benchmark/analysis/output/DLPFC"
ablation_path = "/home/qinxianhan/project/spatial/cadast_demo/output/DLPFC/ablation"
conditions = os.listdir(ablation_path)


def load_meta(sample):
    meta = pd.read_csv(f"{meta_path}/{sample}/CadaST/metadata.tsv", sep="\t", index_col=0)
    for condition in conditions:
        ab_clusters = pd.read_csv(f"{ablation_path}/{condition}/{sample}_clusters.csv", index_col=0)
        meta[condition] = ab_clusters["domain"].astype("category")
    meta["full"] = meta["domain"]
    meta.dropna(inplace=True)
    return meta


for i, sample in enumerate(samples):
    meta = load_meta(sample)
    meta = meta.astype("category")
    adatas[i] = adatas[i][meta.index]
    # Drop overlapping columns from meta before joining
    meta_to_join = meta.drop(columns=["in_tissue", "array_row", "array_col"])
    adatas[i].obs = adatas[i].obs.join(meta_to_join)


conditions = ["baseline", "fs", "swa", "fs_swa", "full"]


def get_metric(adata, metric=adjusted_rand_score):
    metrics = {}
    truth = adata.obs["truth"] if "truth" in adata.obs else adata.obs["sce.layer_guess"]
    for condition in conditions:
        pred = adata.obs[condition]
        metrics[condition] = metric(truth, pred)
    return metrics


get_metric(adatas[0])

nmi_df = pd.DataFrame(columns=conditions, index=samples)
nmis = [get_metric(adata, metric=normalized_mutual_info_score) for adata in adatas]
for i, sample in enumerate(samples):
    nmi_df.loc[sample] = nmis[i]
nmi_df


ari_df = pd.DataFrame([get_metric(adata) for adata in adatas], index=samples)
ari_df

hom_df = pd.DataFrame([get_metric(adata, metric=homogeneity_score) for adata in adatas], index=samples)
hom_df

adatas[0].obsm["spatial"]

# +


def get_silhouette(adata):
    silhouette_scores = {}
    preds = conditions
    for pred in preds:
        labels = adata.obs[pred].values
        d = squareform(pdist(adata.obsm["spatial"], metric="euclidean"))
        silhouette_scores[pred] = silhouette_score(d, labels, metric="precomputed")
    return silhouette_scores


# -

silhouette_df = pd.DataFrame([get_silhouette(adata) for adata in adatas], index=samples)
aws_df = (silhouette_df - silhouette_df.min()) / (silhouette_df.max() - silhouette_df.min())
aws_df

# +

from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler


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


def compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)


# -


def get_chaos(adata):
    chaos = {}
    preds = conditions
    for pred in preds:
        labels = adata.obs[pred].values
        chaos[pred] = compute_CHAOS(labels, adata.obsm["spatial"])
    return chaos


def get_pas(adata):
    pas = {}
    preds = conditions
    for pred in preds:
        labels = adata.obs[pred].values
        pas[pred] = compute_PAS(labels, adata.obsm["spatial"])
    return pas


chaos_df = pd.DataFrame([get_chaos(adata) for adata in adatas], index=samples)
pas_df = pd.DataFrame([get_pas(adata) for adata in adatas], index=samples)
chaos_df
pas_df

chaos_df_r = 1 - chaos_df
chaos_scaled = (chaos_df_r - chaos_df_r.min()) / (chaos_df_r.max() - chaos_df_r.min())
pas_df_r = 1 - pas_df
pas_scaled = (pas_df_r - pas_df_r.min()) / (pas_df_r.max() - pas_df_r.min())

# +
ari_mean = ari_df.mean()
nmi_mean = nmi_df.mean()
hom_mean = hom_df.mean()
chaos_mean = chaos_scaled.mean()
pas_mean = pas_scaled.mean()
aws_mean = aws_df.mean()

metrics_df = pd.DataFrame(
    {"ARI": ari_mean, "NMI": nmi_mean, "HOM": hom_mean, "CHAOS": chaos_mean, "PAS": pas_mean, "AWS": aws_mean}
).T
conditions = ["baseline", "fs", "swa", "fs_swa", "full"]
conditions_r = ["full", "fs_swa", "swa", "fs", "baseline"]
metrics_df = metrics_df[conditions_r]
metrics_df

# -


def plt_table(plot_df: pd.DataFrame, score_cols: list, acc_cols: list, coh_cols: list):
    from plottable.table import Table
    from plottable import ColumnDefinition
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    import matplotlib as mpl

    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
    cmap_fn_bar = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.YlGnBu, num_stds=2.5)

    column_definitions = [
        ColumnDefinition("Method", width=0.6, textprops={"ha": "left", "weight": "medium"}),
    ]
    # Circles for the acc values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=0.5,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.2}, "weight": "medium"},
            cmap=cmap_fn(plot_df[col]),
            formatter="{:.2f}",
            group="Prediction Accuracy",
        )
        for i, col in enumerate(acc_cols)
    ]
    # Circles for the coherence values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=0.5,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.2}, "weight": "medium"},
            cmap=cmap_fn(plot_df[col]),
            formatter="{:.2f}",
            group="Spatial Coherence",
        )
        for i, col in enumerate(coh_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=0.5,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": mpl.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
                "textprops": {"weight": "medium", "fontsize": 11},
            },
            border="left" if i == 0 else None,
            group="Aggregate Score",
        )
        for i, col in enumerate(score_cols)
    ]
    # Allow to manipulate text post-hoc (in illustrator)
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(plot_df.columns) * 1.2, 2 + 0.6 * plot_df.shape[0]))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 11, "ha": "center", "weight": "medium"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
            # group_label_kw={"weight": "bold"}  # Add bold weight to the group labels
        ).autoset_fontcolors(colnames=plot_df.columns)
    return fig


acc = ["ARI", "NMI", "HOM"]
coh = ["CHAOS", "PAS", "AWS"]
plot_df = metrics_df.T
plot_df["Accuracy"] = plot_df[acc].mean(axis=1)
plot_df["Coherence"] = plot_df[coh].mean(axis=1)
plot_df["Total"] = plot_df[["Accuracy", "Coherence"]].mean(axis=1)
plot_df["Method"] = plot_df.index
# plot_df = plot_df.sort_values(by="Total", ascending=False)
fig = plt_table(plot_df, score_cols=["Accuracy", "Coherence", "Total"], acc_cols=acc, coh_cols=coh)
fig.savefig(
    "/home/qinxianhan/project/spatial/cadast_demo/figures/DLPFC/ablation_table.pdf", bbox_inches="tight", dpi=300
)

# +
ari_mean = ari_df.mean()
nmi_mean = nmi_df.mean()
hom_mean = hom_df.mean()
chaos_mean = chaos_df.mean()
pas_mean = pas_df.mean()
aws_mean = silhouette_df.mean()

metrics_df = pd.DataFrame(
    {"ARI": ari_mean, "NMI": nmi_mean, "HOM": hom_mean, "CHAOS": chaos_mean, "PAS": pas_mean, "AWS": aws_mean}
).T
conditions = ["baseline", "fs", "swa", "fs_swa", "full"]
conditions_r = ["full", "fs_swa", "swa", "fs", "baseline"]
metrics_df = metrics_df[conditions_r]
metrics_df

# -


def plt_table(plot_df: pd.DataFrame, score_cols: list, acc_cols: list, coh_cols: list):
    from plottable.table import Table
    from plottable import ColumnDefinition
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    import matplotlib as mpl

    cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
    cmap_fn_bar = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.YlGnBu, num_stds=2.5)

    column_definitions = [
        ColumnDefinition("Method", width=0.6, textprops={"ha": "left", "weight": "medium"}),
    ]
    # Circles for the acc values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=0.5,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.2}, "weight": "medium"},
            cmap=cmap_fn(plot_df[col]),
            formatter="{:.2f}",
            group="Prediction Accuracy",
        )
        for i, col in enumerate(acc_cols)
    ]
    # Circles for the coherence values
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=0.5,
            textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.2}, "weight": "medium"},
            cmap=cmap_fn(plot_df[col]),
            formatter="{:.2f}",
            group="Spatial Coherence",
        )
        for i, col in enumerate(coh_cols)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=0.5,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": mpl.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
                "textprops": {"weight": "medium", "fontsize": 11},
            },
            border="left" if i == 0 else None,
            group="Aggregate Score",
        )
        for i, col in enumerate(score_cols)
    ]
    # Allow to manipulate text post-hoc (in illustrator)
    with mpl.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(plot_df.columns) * 1.2, 2 + 0.6 * plot_df.shape[0]))
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 11, "ha": "center", "weight": "medium"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
            # group_label_kw={"weight": "bold"}  # Add bold weight to the group labels
        ).autoset_fontcolors(colnames=plot_df.columns)
    return fig


acc = ["ARI", "NMI", "HOM"]
coh = ["CHAOS", "PAS", "AWS"]
plot_df = metrics_df.T
plot_df["Accuracy"] = plot_df[acc].mean(axis=1)
plot_df["Coherence"] = plot_df[coh].mean(axis=1)
plot_df["Total"] = plot_df[["Accuracy", "Coherence"]].mean(axis=1)
plot_df["Method"] = plot_df.index
fig = plt_table(plot_df, score_cols=["Accuracy", "Coherence", "Total"], acc_cols=acc, coh_cols=coh)
fig.savefig(
    "/home/qinxianhan/project/spatial/cadast_demo/figures/DLPFC/ablation_table_raw.pdf", bbox_inches="tight", dpi=300
)
