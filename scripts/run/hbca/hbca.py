import os
from anndata import read_h5ad
from datetime import date
from cadaST import CadaST

today = date.today().strftime("%m%d")

beta, alpha, theta, init_alpha = 10, 0.6, 0.15, 6
max_iter = 3
kneighbors = 18
NPROCESS = 16
sample = "hbca"

dataPath = f"../dataset/visium_hbca/{sample}.h5ad"
n_top = 2000


def main():
    adata = read_h5ad(dataPath)
    print(adata.shape)
    path = f"../output/{sample}/{today}/"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}log.txt", "w") as f:
        f.write(
            f"{dataPath}, \nbeta={beta}, alpha={alpha}, theta={theta}, init_alpha={init_alpha}, max_iter={max_iter}, kneighbors={kneighbors}, n_top={n_top}\n"
        )
    model = CadaST(
        adata=adata, kneighbors=kneighbors, alpha=alpha, theta=theta, max_iter=max_iter, n_top=n_top, n_jobs=32
    )
    print("Start running:")
    adata_processed = model.fit()
    adata_processed.write(f"{path}adata.h5ad")


if __name__ == "__main__":
    main()
