# %%
import os
import scanpy as sc
import matplotlib.pyplot as plt
from params import init_plot_params

init_plot_params()
data_path = "/home/qinxianhan/project/spatial/cadast_demo/output/ATAC"
plt_path = "/home/qinxianhan/project/spatial/cadast_demo/figure/ATAC"


# %%
def load_adata(file_dir) -> sc.AnnData:
    adata = sc.read_h5ad(file_dir, backed="r")
    print(adata)
    return adata


def domain_plt(adata, sample, **kwargs):
    plt.figure(figsize=(8, 8))
    plt_clr = "domain" if "domain" in adata.obs else "leiden"
    plt_output = os.path.join(plt_path, sample + "_domain.pdf")
    sc.pl.embedding(adata, basis="spatial", color=plt_clr, show=False, **kwargs)
    plt.savefig(plt_output, bbox_inches="tight")
    plt.close()
    print(f"Plot domain for {sample} saved to {plt_output}")


def marker_plt(adata, sample, markers, **kwargs):
    for marker in markers:
        plt.figure(figsize=(8, 8))
        plt_output = os.path.join(plt_path, sample + f"_{marker}.pdf")
        sc.pl.embedding(adata, basis="spatial", color=marker, show=False, cmap="virdis", **kwargs)
        plt.savefig(plt_output, bbox_inches="tight")
        plt.close()
        print(f"Plot marker {marker} for {sample} saved to {plt_output}")


# %%
def main():
    samples_dir = os.listdir(data_path)
    sizes = (25, 100)
    title = "CadaST"
    for i, sample_dir in enumerate(samples_dir):
        file_dir = os.path.join(data_path, sample_dir)
        adata = load_adata(file_dir)
        sample = sample_dir.split("_")[0]
        domain_plt(adata, sample, size=sizes[i], frameon=False, alpha=0.9, title=title)


if __name__ == "__main__":
    main()

# %%
