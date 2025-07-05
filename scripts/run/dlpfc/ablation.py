import scanpy as sc
import os
from cadaST import CadaST
from cadaST.utils import clustering

params_set = {
    "baseline": {
        "beta": 10,
        "alpha": 0.6,
        "theta": 0.2,
        "init_alpha": 6,
        "max_iter": 0,
        "icm_iter": 0,
        "kneighbors": 18,
        "n_top": None,
    },
    "fs": {
        "beta": 10,
        "alpha": 0.6,
        "theta": 0.2,
        "init_alpha": 6,
        "max_iter": 0,
        "icm_iter": 0,
        "kneighbors": 18,
        "n_top": 2000,
    },
    "swa": {
        "beta": 10,
        "alpha": 0.6,
        "theta": 1,
        "init_alpha": 6,
        "max_iter": 3,
        "icm_iter": 0,
        "kneighbors": 18,
        "n_top": None,
    },
    "fs_swa": {
        "beta": 10,
        "alpha": 0.6,
        "theta": 1,
        "init_alpha": 6,
        "max_iter": 3,
        "icm_iter": 0,
        "kneighbors": 18,
        "n_top": 2000,
    },
}

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

data_path = "/home/qinxianhan/project/spatial/cadast_demo/dataset/DLPFC"
out_path = "/home/qinxianhan/project/spatial/cadast_demo/output/DLPFC/ablation"
os.makedirs(out_path, exist_ok=True)


def run_ablation(sampleid, params: dict, output_path: str):
    adata = sc.read_h5ad(os.path.join(data_path, f"{sampleid}_processed.h5ad"))
    n_clusters = len(adata.obs["truth"].unique())
    print(sampleid)
    os.makedirs(output_path, exist_ok=True)
    model = CadaST(
        adata=adata,
        kneighbors=params["kneighbors"],
        beta=params["beta"],
        alpha=params["alpha"],
        theta=params["theta"],
        init_alpha=params["init_alpha"],
        max_iter=params["max_iter"],
        icm_iter=params["icm_iter"],
        n_top=params["n_top"],
        n_jobs=32,
    )
    adata = model.fit()
    clustering(adata, n_clusters=n_clusters)
    adata.obs.to_csv(os.path.join(output_path, f"{sampleid}_clusters.csv"))


def main():
    for name, params in params_set.items():
        print(f"Running ablation for {name} with params: {params}")
        output_path = os.path.join(out_path, name)
        os.makedirs(output_path, exist_ok=True)
        for sampleid in sampleids:
            run_ablation(sampleid, params, output_path)


def test():
    sampleid = "151507"
    params = params_set["baseline"]
    output_path = os.path.join(out_path, "test")
    adata = sc.read_h5ad(os.path.join(data_path, f"{sampleid}_processed.h5ad"))
    print(adata)
    run_ablation(sampleid, params, output_path)


main()