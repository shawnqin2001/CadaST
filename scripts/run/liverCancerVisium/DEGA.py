import os

import cadaST
import numpy as np
import pandas as pd
import scanpy as sc

date = "1019"
dataPath = "/home/qinxianhan/project/spatial/cadast_demo/dataset/visium_liverCancer"
rawPath = "/home/qinxianhan/project/spatial/dataset/LCVisium/preprocessed"
outputPath = (
    f"/home/qinxianhan/project/spatial/cadast_demo/output/visium_liverCancer/{date}"
)


def read_raw(samples):
    path = os.path.join(rawPath, samples)
    adata_raw = sc.read_visium(path)
    adata_raw.var_names_make_unique()
    sc.pp.normalize_total(adata_raw, target_sum=1e4)
    sc.pp.log1p(adata_raw)
    sc.pp.scale(adata_raw)
    return adata_raw


def run_model(adata_raw):
    alpha = 0.6
    theta = 0.2
    init_alpha = 6
    max_iter = 1
    icm_iter = 3
    kneighbors = 10
    beta = 10
    n_top = 2000
    model = cadaST.CadaST(
        adata_raw,
        alpha=alpha,
        theta=theta,
        init_alpha=init_alpha,
        beta=beta,
        max_iter=max_iter,
        icm_iter=icm_iter,
        kneighbors=kneighbors,
        n_top=n_top,
        n_jobs=16,
    )
    adata_fit = model.fit()
    return adata_fit.var_names, adata_fit.layers["labels"]


def iou_score(domain_bool, pred_labels):
    intersction = np.logical_and(domain_bool[:, None], pred_labels).sum(0)
    union = np.logical_or(domain_bool[:, None], pred_labels).sum(0)
    iou = intersction / union
    return iou


def main():
    samples = os.listdir(dataPath)
    samples = [i.split(".")[0] for i in samples]
    samples = [sample for sample in samples if sample.endswith("L")]
    iou_genes = {}
    for sample in samples:
        adata = sc.read_h5ad(os.path.join(outputPath, sample, f"{sample}_out.h5ad"))
        domain_bool = np.array(adata.obs["domain"] == 4)
        adata_raw = read_raw(sample)
        genes_list, pred_labels = run_model(adata_raw)
        ious = iou_score(domain_bool, pred_labels)
        max_idx = np.argsort(ious)[::-1][:100]
        max_genes = genes_list[max_idx]
        iou_genes[sample] = max_genes

    df = pd.DataFrame(iou_genes)
    df.to_csv("./iou_genes.csv")


if __name__ == "__main__":
    main()
