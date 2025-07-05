import os
from datetime import date
from anndata import read_h5ad
from cadaST import CadaST

today = date.today().strftime("%m%d")
beta, alpha, theta, init_alpha = 10, 0.4, 0.2, 6
max_iter = 2
kneighbors = 18
sample = "DLPFC"
sampleids = [
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
n_top = 2000

data_path = "/home/qinxianhan/project/spatial/cadast_demo/dataset/DLPFC"


def main():
    for sampleid in sampleids:
        adata = read_h5ad(f"{data_path}/{sampleid}_processed.h5ad")
        print(sampleid)
        print(adata.shape)
        path = f"../output/DLPFC/{today}/{sampleid}/"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}log.txt", "w") as f:
            f.write(
                f"beta={beta}, alpha={alpha}, theta={theta}, init_alpha={init_alpha}, max_iter={max_iter}, kneighbors={kneighbors}, sample={sample}, n_top={n_top}\n"
            )
        model = CadaST(
            adata=adata,
            kneighbors=kneighbors,
            beta=beta,
            alpha=alpha,
            theta=theta,
            max_iter=max_iter,
            n_top=n_top,
            n_jobs=16,
        )
        adata = model.fit()
        adata.write(f"{path}adata.h5ad")  # type: ignore


if __name__ == "__main__":
    main()
