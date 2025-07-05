# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cadaST import CadaST
from cadaST.utils import mclust_R
sc.set_figure_params(dpi_save= 300, vector_friendly=True)
sns.set_theme(style="white", font="Arial")

adata_path = "../dataset/MOB/MOBstereo_HV.h5ad"
adata_all_path = "../dataset/MOB/MOBstereo.h5ad"
impute_path = "../output/MOB/0426/a0.8_t0.2_i3_scale/imputed_exp.feather"
label_path = "../output/MOB/0426/a0.8_t0.2_i3_scale/labels.feather"
adata_raw  = sc.read(adata_all_path)
adata = sc.read_h5ad("/home/qinxianhan/project/spatial/cadast_demo/output/MOB/0918/output.h5ad")

plt.rcParams['figure.figsize'] = [5,4]
sc.pl.embedding(adata, basis="spatial", color = "domain", title="CadaST", show=False, size=12 )
plt.axis("off")
plt.tight_layout()
plt.savefig("../figure/MOB/cluster.pdf", dpi=300, bbox_inches='tight')
plt.show()

mclust_R(adata, num_cluster=10)
plt.rcParams['figure.figsize'] = [5,4]
sc.pl.embedding(adata, basis="spatial", color = "mclust", title="CadaST", show=False, size=12 )
plt.axis("off")
plt.tight_layout()
plt.show()

model = CadaST(adata_raw, kneighbors=16, beta = 10, n_top=2000)
adata_fit = model.fit()
adata_fit.obs["domain"] = adata.obs["domain"]
res = adata_fit.uns["rank_genes_groups"]["names"]
adata.obs["domain"] = adata.obs["domain"].astype(str)

adata_tmp = adata.copy()
adata_raw = adata_raw[ :, adata_tmp.var_names]
adata_tmp.X = adata_raw.X

adata_tmp

sc.tl.rank_genes_groups(adata_tmp, groupby="annot", method="wilcoxon")

# +
sns.set_theme(style="white")
# 绘制主图并返回绘图对象
mat = sc.pl.rank_genes_groups_matrixplot(
    adata_tmp,
    n_genes=3,
    values_to_plot="scores",
    # cmap="RdBu_r",
    dendrogram=False,          # 显示聚类树
    figsize=(9, 3),           # 设置图形大小
    show=False,
    colorbar_title="Score",   # 修改颜色条标题
    vmin=-70,
    vmax=70
)

# 删除默认的颜色条
cbar_ax = mat['color_legend_ax']
cbar_ax.remove()

# 获取主图所在的 figure
fig = mat['mainplot_ax'].figure

# 在 figure 中新建一个独立的颜色条 Axes
cbar_ax_new = fig.add_axes([0.77, 0.3, 0.02, 0.4])  # [x, y, width, height]，根据需求调整

# 创建颜色条
cbar = fig.colorbar(
    mat['mainplot_ax'].collections[0],  # 绑定主图的颜色集合
    cax=cbar_ax_new,                    # 使用新建的轴绘制颜色条
    orientation='vertical',             # 设置颜色条方向为竖直

)

# 设置颜色条标题
cbar.set_label("Raw Score", rotation=270, labelpad=15, fontweight='bold')  # 旋转标题为垂直

# 调整布局，显示最终结果
# plt.tight_layout()
plt.savefig("../figure/MOB/matrixRawPlot.pdf", dpi=300, bbox_inches='tight')
# plt.show()
# -

adata_tmp.obs.index

adata_label = adata.copy()
adata_label.X = adata_label.layers["label"]

sc.pp.pca(adata_label, n_comps=20)
mclust_R(adata_label, num_cluster=7)

adata_label = sc.read_h5ad("/home/qinxianhan/project/spatial/cadast_demo/output/MOB/0918/output_label.h5ad")
plt.rcParams['figure.figsize'] = [5,4]
sc.pl.embedding(adata_label, basis="spatial", color = "mclust", show=False, size=10)
plt.axis("off")
plt.title("Binary Clustering", fontsize=15, weight = 'medium')
plt.tight_layout()
# plt.savefig("../figure/MOB/clusterBinary.pdf", dpi=300, bbox_inches='tight')

adata_label.write("/home/qinxianhan/project/spatial/cadast_demo/output/MOB/0918/output_label.h5ad")

sc.tl.rank_genes_groups(adata, groupby="domain")

res = adata.uns["rank_genes_groups"]['names']

res['6']

plt.rcParams['figure.figsize'] = [5,4]
sns.set_theme()
color = res['7'][7]
sc.pl.embedding(adata, basis="spatial", color = color, title="cadaST", color_map = "viridis", show=False, size=30,layer="label" )
plt.axis("off")
plt.tight_layout()
# plt.savefig("../figure/MOB/cluster.png", dpi=300, bbox_inches='tight')
plt.show()

genelist = ["Ptn", "Cck",  "Stmn2","Sox11"]

fig, axs = plt.subplots(1,4, figsize=(20, 4))
for i in range(4):
    sc.pl.embedding(adata, basis="spatial", color = genelist[i], ax=axs[i], show=False, size=10, colorbar_loc=None,
                    cmap='viridis')
    axs[i].axis("off")
fig.tight_layout()
# fig.savefig("../figure/MOB/markerGenes.pdf", dpi=300, bbox_inches='tight')

adata_raw = sc.read_h5ad("../dataset/MOB/MOBstereo_HV.h5ad")
kneighbors = 20
beta = 10
alpha = 0.8
theta = 0.2
init_alpha = 6
graph = SimilarityGraph(adata_raw, kneighbors=kneighbors, beta=beta, alpha=alpha, theta = theta, init_alpha=init_alpha, max_iter=5)
adata.obs["labels"] = graph.labels
sc.pl.embedding(adata, basis="spatial", color = "labels", show=False, size=8)

geneLabelList = [gene+'Label' for gene in genelist]
for geneLabel in geneLabelList:
    adata.obs[geneLabel] = adata.obs[geneLabel].astype("category")

fig, axs = plt.subplots(1,4, figsize=(20, 4))
for i in range(4):
    sc.pl.embedding(adata, basis="spatial", color = geneLabelList[i], ax=axs[i], show=False, size=10, legend_loc=None, palette="viridis")
    axs[i].set_title(None)
    axs[i].axis("off")
fig.tight_layout()
# fig.savefig("../figure/MOB/markerGenesLabels.pdf", dpi=300, bbox_inches='tight')

# +
plt.rcParams['image.cmap'] = 'viridis'
# genelist = ["Ptn", "Cck",  "Stmn2","Sox11"]
genelist = ["Ptn", "Cck", "Sox11"]
geneLabelList = [gene+'Label' for gene in genelist]
fig, axs = plt.subplots(3, 3, figsize=(8,6))
row_labels = ["Raw", "Predicted", "Denoised"]
plt.subplots_adjust(wspace=0.01, hspace=0.01)
# sns.set_theme(color_codes='b')
for i, gene in enumerate(genelist):
    sc.pl.embedding(adata_raw, basis='spatial', color=gene, size=10, ax=axs[0, i], show=False, colorbar_loc=None)
    axs[0,i].axis('off')
    axs[0,i].set_title(gene, fontsize=16)
for i, gene in enumerate(geneLabelList):
    sc.pl.embedding(adata, basis='spatial',color=gene,  size=10, ax=axs[1, i], show=False, colorbar_loc=None, title="", legend_loc=None)
    axs[1, i].axis('off')
for i, gene in enumerate(genelist):
    sc.pl.embedding(adata, basis='spatial',color=gene,  size=10, ax=axs[2, i], show=False, colorbar_loc=None, title="")
    axs[2, i].axis("off")

for i, label in enumerate(row_labels):
    axs[i, 0].text(-0.1, 0.5, label, fontsize=16, ha='right', va='center', transform=axs[i, 0].transAxes, rotation=90)
fig.tight_layout()
fig.savefig("../figure/MOB/markers.pdf", bbox_inches='tight')
# -

staMeta = pd.read_csv("/home/qinxianhan/project/spatial/cadast_demo/benchmark/STAGATE/results/MOB/STAGATE_louvain7.csv", index_col=0)
gstMeta = pd.read_csv("/home/qinxianhan/project/spatial/cadast_demo/benchmark/GraphST/results/MOB/GraphST.csv", index_col=0)
adata.obs["STAGATE"] = staMeta["louvain"].astype("category")
adata.obs["GraphST"] = gstMeta["domain"].astype("category")

# +
sns.set_theme(font_scale=1.2)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
sc.pl.embedding(adata, basis="spatial", color = "STAGATE", ax=axs[0], show=False, frameon = False, size=12, legend_loc=None)
axs[0].set_title("STAGATE", fontsize=18, weight='medium')
sc.pl.embedding(adata, basis="spatial", color = "GraphST", ax=axs[1], show=False, frameon = False, size=12, legend_loc=None)
axs[1].set_title("GraphST", fontsize=18, weight='medium')
sc.pl.embedding(adata, basis="spatial", color = "domain", ax=axs[2], show=False, frameon = False, size=12)
axs[2].set_title("CadaST", fontsize=18, weight='medium')

fig.tight_layout()
# fig.savefig("../figure/MOB/compare.pdf", dpi=300, bbox_inches='tight')
# -

domains = ["GCL", "EPL", "IPL", "GL", "ONL", "MCL", "RMS"]
annoMap = {(i + 1): domain for i, domain in enumerate(domains)}
adata.obs["annot"] = adata.obs["domain"].map(annoMap)

sc.pl.embedding(adata, basis="spatial", color = "annot", show=False, size=12)
plt.axis("off")
plt.show()

# +
domains_orders = ["ONL", "GL", "EPL", "MCL", "IPL", "GCL", "RMS"]
row_labels = ["CadaST", "STAGATE", "GraphST"]
sta_order = [3, 7, 2, 6, 4, 1, 5 ]
graph_order = [6, 7, 1, 5, 4, 2, 3]
fig, axs = plt.subplots(3, 7, figsize = (28,9))
for i, domain in enumerate(domains_orders):
    sc.pl.embedding(adata, basis="spatial", color = "annot", ax=axs[0, i],groups=[domains_orders[i]], show=False,size=12, legend_loc=None,  title="")
    axs[0, i].set_title(domain, fontsize=30)
    axs[0, i].axis("off")
for i, domain in enumerate(sta_order):
    sc.pl.embedding(adata, basis="spatial", color = "STAGATE", ax=axs[1, i], groups=[domain], show=False,size=12 , legend_loc=None, title="")
    axs[1, i].axis("off")
for i, domain in enumerate(graph_order):
    sc.pl.embedding(adata, basis="spatial", color = "GraphST", ax=axs[2, i], groups=[domain], show=False,size=12,  legend_loc=None, title="")
    axs[2, i].axis("off")

for i in range(3):
    axs[i, 0].text(-0.1, 0.5, row_labels[i], fontsize=30, ha='right', va='center', transform=axs[i, 0].transAxes, rotation=90)
fig.tight_layout()
fig.savefig("../figure/MOB/domains.pdf", dpi=300, bbox_inches='tight')


# +

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
    results = [
        fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel)
        for i in range(matched_location.shape[0])
    ]
    return np.sum(results) / len(clusterlabel)



# -

PASs = []
CHASs = []
for domain_num in range(6,11):
    clustering(adata, n_clusters=domain_num, dims = 15)
    PASs.append(compute_PAS(adata.obs["domain"], adata.obsm["spatial"]))
    CHASs.append(compute_CHAOS(adata.obs["domain"], adata.obsm["spatial"]))

CHASs

sns.set_theme()
# barplot CHASs
fig, ax = plt.subplots()
ax.bar(range(6,11), CHASs)
ax.set_xlabel("Number of Domains")
ax.set_ylabel("CHAOS")
fig.show()

stcada = np.load("../output/MOB/CHAOS.npy")
graphst = np.load("/home/qinxianhan/project/spatial/benchmark/GraphST/results/MOB/CHAOS.npy")
stagate = np.load("/home/qinxianhan/project/spatial/benchmark/STAGATE/results/MOB/CHAOS.npy")
scanpy = np.load("/home/qinxianhan/project/spatial/benchmark/STAGATE/results/MOB/scanpy_CHAOS.npy")

CHAOS_df = pd.DataFrame({"cadaST": stcada,  "STAGATE": stagate, "GraphST": graphst})
# CHOAS_df = CHOAS_df.melt(var_name="Method", value_name="CHAOS")
CHAOS_df

CHAOS_df.set_index(pd.Index(range(6,11), name="Domains"), inplace=True)
CHAOS_df

CHAOS_melt = CHAOS_df.melt(var_name="Method", value_name="CHAOS", ignore_index=False)
CHAOS_melt.reset_index(inplace=True)

CHAOS_melt

# +

cusPalette = ["#336469", "#255FA7", "#A194C6", "#5491BB","#F16E65", "#FAAF76" ]
cusPalette
# -

cusPalette = [ "#F16E65", "#336469", "#255FA7" ]

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(8,4))
ax = sns.barplot(data=CHAOS_melt, x = 'Domains', y="CHAOS", hue="Method", palette=cusPalette, width=0.9, alpha=0.9,
            edgecolor=".2", linewidth=3)
ax.set_xlabel("Number of Clusters", fontsize=15, fontweight='bold')
ax.set_ylabel("CHAOS",fontsize=15, fontweight='bold')
ax.set_ylim(0.015, 0.0205)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.legend(title="", loc='lower center', bbox_to_anchor=(0.5, -0.25), ncols=3, frameon=False)
ax.legend(title="", loc='upper left', frameon=False)
fig.tight_layout()
fig.savefig("../figure/MOB/CHAOS.pdf", dpi=300, bbox_inches='tight')
