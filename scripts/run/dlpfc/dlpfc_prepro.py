import scanpy as sc
import pandas as pd


def main():
    dataPath = "/home/qinxianhan/project/spatial/benchmark/analysis/data/DLPFC"
    outPath = "/home/qinxianhan/project/spatial/cadast_demo/dataset/DLPFC"
    annPath = f"{outPath}/annotation"
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

    for sample in sampleids:
        adata = sc.read_visium(f"{dataPath}/{sample}")
        adata.var_names_make_unique()
        adata = data_preprocess(adata, min_cells=10)
        truth = pd.read_csv(
            f"{annPath}/{sample}_truth.txt", index_col=0, sep="\t", header=None
        )
        adata.obs["truth"] = truth.loc[adata.obs.index, 1].values
        adata = adata[~adata.obs["truth"].isna()]
        adata.write_h5ad(f"{outPath}/{sample}_processed.h5ad")


if __name__ == "__main__":
    main()
