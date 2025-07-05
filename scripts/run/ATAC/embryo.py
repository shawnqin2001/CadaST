# %%
import os
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def init_plot_params():
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "medium"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["savefig.bbox"] = "tight"
    rcParams["figure.figsize"] = (8, 8)
    sns.set_theme(context="notebook", style="white", font="Arial", palette="muted")
    sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)


init_plot_params()
data_path = "/home/qinxianhan/project/spatial/cadast_demo/output/ATAC"
fit_out_path = "/home/qinxianhan/project/spatial/cadast_demo/figure/ATAC"


# %%
def load_adata(file_dir) -> sc.AnnData:
    adata = sc.read_h5ad(file_dir)
    # print(adata)
    return adata


samples_dir = os.listdir(data_path)
sample_dir = samples_dir[0]
sample = sample_dir.split("_")[0]
file_dir = os.path.join(data_path, sample_dir)
fig_path = os.path.join(fit_out_path, sample)
adata = load_adata(file_dir)
print(adata.shape)
print(fig_path)

# %%
plt_domain = "domain" if "domain" in adata.obs else "leiden"
sc.pl.embedding(adata, basis="spatial", color=plt_domain, frameon=False, alpha=0.9, size=30)

# %%
sc.tl.rank_genes_groups(adata, groupby=plt_domain, n_genes=8, method="wilcoxon")
sc.pl.rank_genes_groups(
    adata,
    n_genes=8,
)
# %%
markers = adata.uns["rank_genes_groups"]["names"]
# %%
for target_domain in adata.obs[plt_domain].unique():
    target_domain = str(target_domain)
    sc.pl.embedding(
        adata,
        basis="spatial",
        color=markers[target_domain],
        frameon=False,
        alpha=0.9,
        size=30,
        cmap="viridis",
        ncols=len(markers[target_domain]) // 2,
        show=False,
    )
    plt.savefig(os.path.join(fig_path, f"{target_domain}_markers.pdf"), bbox_inches="tight")
    plt.close()
# %%
