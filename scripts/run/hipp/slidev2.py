import os
from anndata import read_h5ad
from datetime import date
from cadaST import *

today = date.today().strftime("%m%d")
rep = ""
today = today + rep
beta, alpha, theta, init_alpha = 10, 0.8, 0.2, 6
max_iter = 3
kneighbors = 32
n_jobs = 16
dataPath = "/home/qinxianhan/project/spatial/cadast_demo/dataset/slide_hipp/preprocessed2.h5ad"

n_top = 2000


def main():
    adata = read_h5ad(dataPath)
    print(adata.shape)
    path = f"../../output/slide/{today}/"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}log.txt", "w") as f:
        f.write(
            f"beta={beta}, alpha={alpha}, theta={theta}, init_alpha={init_alpha}, max_iter={max_iter}, kneighbors={kneighbors}, n_top={n_top}\n"
        )
    model = CadaST(adata, kneighbors=kneighbors, beta=beta, alpha=alpha, theta=theta, init_alpha=init_alpha, n_top=n_top, n_jobs=n_jobs) 
    adata = model.fit()
    adata.write(f"{path}slide.h5ad")


if __name__ == "__main__":
    main()
