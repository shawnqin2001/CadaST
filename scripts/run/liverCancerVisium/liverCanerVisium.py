import os
from anndata import read_h5ad
from datetime import date
from cadaST import CadaST

today = date.today().strftime("%m%d")

beta, alpha, theta, init_alpha = 10, 0.6, 0.2, 6
max_iter = 2
kneighbors = 18
n_jobs = 24
n_top = 2000

dataPath = "/home/qinxianhan/project/spatial/cadast_demo/dataset/visium_liverCancer"
outputPath = (
    f"/home/qinxianhan/project/spatial/cadast_demo/output/visium_liverCancer/{today}/"
)
samples = os.listdir(dataPath)
samples = [sample.removesuffix(".h5ad") for sample in samples]
print(samples)


def main():
    for sample in samples:
        output = os.path.join(outputPath, sample)
        os.makedirs(output, exist_ok=True)
        with open(f"{output}/log.txt", "w") as f:
            f.write(
                f"alpha={alpha}, theta={theta}, init_alpha={init_alpha}, max_iter={max_iter}, kneighbors={kneighbors}, sample={sample}, n_top={n_top}\n"
            )
        adata = read_h5ad(f"{dataPath}/{sample}.h5ad")
        print(adata.shape)
        model = CadaST(
            adata=adata,
            kneighbors=kneighbors,
            alpha=alpha,
            theta=theta,
            max_iter=max_iter,
            n_top=n_top,
            n_jobs=n_jobs,
        )
        print(f"Start running:{sample}")
        model.construct_graph()
        adata = model.fit()
        adata.write(filename=f"{output}/{sample}.h5ad")  # type: ignore


if __name__ == "__main__":
    main()
