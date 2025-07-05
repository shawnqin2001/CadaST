import os
from anndata import read_h5ad
from cadaST import *
from datetime import date

today = date.today().strftime("%m%d")

beta, alpha, theta, init_alpha = 10, 0.8, 0.2, 6
kneighbors = 18
max_iter = 3
n_top = 2000
icm_iter = 1
sample = "MOB"


def main():
    adata = read_h5ad(f"../dataset/MOB/MOBstereo.h5ad")
    print(adata.shape)
    path = f"../output/{sample}/{today}/a{alpha}_t{theta}_i{max_iter}_scale/"
    os.makedirs(path, exist_ok=True)
    print("Start running with:")
    print(f"Sample: {sample}")
    model = CadaST(
        adata,
        kneighbors=kneighbors,
        beta=beta,
        alpha=alpha,
        theta=theta,
        init_alpha=init_alpha,
        n_top=n_top,
        NPROCESS=6,
    )
    adata_fit = model.fit()
    adata_fit.write(f"{path}MOBstereo.h5ad")


if __name__ == "__main__":
    main()
