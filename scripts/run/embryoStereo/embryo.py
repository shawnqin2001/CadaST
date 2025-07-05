import os
from anndata import read_h5ad
from datetime import date
from cadaST import CadaST, clustering
import scanpy as sc

today = date.today().strftime("%m%d")
beta, alpha, theta, init_alpha = 10, 0.6, 0.2, 6
max_iter = 2
kneighbors = 16
n_jobs = 16
scale = False
samples = ["E13.5", "E14.5"]
dataPaths = [
    "/home/qinxianhan/project/spatial/dataset/MouseEmbryo/E13.5.h5ad",
    "/home/qinxianhan/project/spatial/cadast_demo/dataset/embryo/E14.5_E1S3.MOSTA.h5ad",
]

n_top = 2000


def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def main(sample, dataPath):
    adata = read_h5ad(dataPath)
    print(adata.shape)
    adata = preprocess(adata)
    if scale:
        sc.pp.scale(adata)
    path = f"../../output/embryo/{today}/{sample}/"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}log.txt", "w") as f:
        f.write(
            f"{dataPath}, \nbeta={beta}, alpha={alpha}, theta={theta}, init_alpha={init_alpha}, max_iter={max_iter}, kneighbors={kneighbors}, scale = {scale}\n"
        )
    model = CadaST(
        adata=adata,
        kneighbors=kneighbors,
        alpha=alpha,
        theta=theta,
        max_iter=max_iter,
        n_top=n_top,
        n_jobs=n_jobs,
    )
    print("Start running:")
    model.construct_graph()
    adata = model.fit()
    clustering(adata, n_clusters=18)
    adata.write(f"{path}adata.h5ad")


if __name__ == "__main__":
    for sample, dataPath in zip(samples, dataPaths):
        main(sample, dataPath)
