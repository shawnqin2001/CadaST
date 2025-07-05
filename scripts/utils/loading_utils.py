import os
import pandas as pd
import numpy as np
import anndata


def load_her2(dataset_path, section_id="A1"):
    # load HER2 dataset
    dataset_path = os.path.join(dataset_path, "HER2")
    cm_path = os.path.join(dataset_path, "count-matrices")
    meta_path = os.path.join(dataset_path, "meta")
    cm_file = section_id + ".tsv"
    meta_file = section_id + "_labeled_coordinates.tsv"
    cm = pd.read_csv(os.path.join(cm_path, cm_file), sep="\t", header=0, index_col=0)
    meta = pd.read_csv(os.path.join(meta_path, meta_file), sep="\t", header=0)
    keep_cells = meta.dropna().index
    meta = meta.iloc[keep_cells]
    xs = meta["x"].tolist()
    ys = meta["y"].tolist()

    rounded_xs = [round(x) for x in xs]
    rounded_ys = [round(y) for y in ys]

    coord = [str(x) + "x" + str(y) for x, y in zip(rounded_xs, rounded_ys)]
    meta["Row.names"] = coord
    st_X = cm.to_numpy()
    meta.sort_values(by="Row.names", inplace=True)
    meta = meta.loc[meta["Row.names"].isin(cm.index)]
    meta = meta.reset_index(drop=True)

    var_df = pd.DataFrame(cm.columns, columns=["Genes"])
    adata = anndata.AnnData(X=st_X, obs=meta, var=var_df)
    adata.obs["Ground_Truth"] = adata.obs["label"]
    spatial = np.vstack((adata.obs["pixel_x"].to_numpy(), adata.obs["pixel_y"].to_numpy()))
    adata.obsm["spatial"] = spatial.T
    return adata
