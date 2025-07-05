import os
from anndata import read_h5ad
from datetime import date
from cadaST import CadaST
import time
import scanpy as sc
import pandas as pd

today = date.today().strftime("%m%d")
beta, alpha, theta, init_alpha = 10, 0.6, 0.2, 6
max_iter = 2
kneighbors = 16
n_jobs = 50
scale = False
dataPath = "/home/qinxianhan/project/spatial/dataset/MouseEmbryo"
n_top = 2000


def process(sample, results: pd.DataFrame, reps=5):
    datadir = f"{dataPath}/{sample}.h5ad"
    path = f"../../output/embryo/{today}/{sample}/"
    os.makedirs(path, exist_ok=True)
    adata = read_h5ad(datadir)
    print(adata.shape)
    print("Preprocessing adata")
    if scale:
        sc.pp.scale(adata)
    for rep in range(reps):
        print(f"Processing {sample} rep {rep}")
        model = CadaST(
            adata,
            beta=beta,
            alpha=alpha,
            theta=theta,
            init_alpha=init_alpha,
            max_iter=max_iter,
            kneighbors=kneighbors,
            n_jobs=n_jobs,
            n_top=n_top,
        )
        start_time = time.time()
        model.construct_graph()
        model.fit()
        time_consume = time.time() - start_time
        results.loc[sample, f"rep{rep}"] = time_consume


def main():
    reps = 5
    samples = [
        "E9.5.h5ad",
        "E10.5.h5ad",
        "E11.5.h5ad",
        "E12.5.h5ad",
        "E13.5.h5ad",
        "E14.5.h5ad",
        "E15.5.h5ad",
        "E16.5.h5ad",
    ]
    samples = [os.path.splitext(sample)[0] for sample in samples]
    results = pd.DataFrame(columns=[f"rep{i}" for i in range(reps)], index=samples)
    for sample in samples:
        print(f"Processing {sample}")
        process(sample, results, reps=reps)
    results.to_csv("time_reps.csv")


if __name__ == "__main__":
    main()
