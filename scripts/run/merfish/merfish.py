import os
from anndata import read_h5ad
from src.model import MultiGeneGraph
from datetime import date

today = date.today().strftime("%m%d")

beta, alpha, theta, init_alpha = 10, 0.6, 0.4, 6
max_iter = 5
kneighbors = 18
dataPath = "./dataset/merfish/processed.h5ad"
n_top = None


def main():
    adata = read_h5ad(dataPath)
    print(adata.shape)
    exp = adata.to_df()
    coord = adata.obsm["spatial"]
    path = f"./output/merfish/{today}/"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}log.txt", "w") as f:
        f.write(
            f"beta={beta}, alpha={alpha}, theta={theta}, init_alpha={init_alpha}, max_iter={max_iter}, kneighbors={kneighbors}, n_top={n_top}\n"
        )
    model = MultiGeneGraph(
        exp_matrix=exp,
        coord=coord,
        kneighbors=kneighbors,
        alpha=alpha,
        theta=theta,
        max_iter=max_iter,
        n_top=n_top,
        NPROCESS=8,
    )
    save = {
        "exp": f"{path}imputed_exp2.feather",
        "labels": f"{path}labels2.feather",
    }
    print("Start running:")
    print(f"beta = {beta}, alpha= {alpha}, theta={theta}, numIteration = {max_iter}")
    model.fit(save=save)


if __name__ == "__main__":
    main()
