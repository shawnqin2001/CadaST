import numpy as np
import scanpy as sc
import seaborn as sns
from scipy.sparse import csr_matrix
from matplotlib import rcParams

from .graph import SimilarityGraph


def init_plot_params():
    rcParams["font.family"] = "Arial"
    rcParams["font.weight"] = "medium"
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    sns.set_theme(style="white", font="Arial")
    sc.set_figure_params(vector_friendly=True, dpi=96, dpi_save=300)


def mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="X_pca",
    random_seed=2024,
    verbose=False,
):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import rpy2.robjects.numpy2ri
    import rpy2.robjects as robjects

    np.random.seed(random_seed)

    robjects.r.library("mclust")

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]
    if not verbose:
        import contextlib
        from io import StringIO

        with (
            contextlib.redirect_stdout(StringIO()),
            contextlib.redirect_stderr(StringIO()),
        ):
            res = rmclust(
                rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]),
                num_cluster,
                modelNames,
            )
    else:
        res = rmclust(
            rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]),
            num_cluster,
            modelNames,
        )
    mclust_res = np.array(res[-2])

    adata.obs["mclust"] = mclust_res
    adata.obs["mclust"] = adata.obs["mclust"].astype("int").astype("category")

    return adata


def lap_score(X, W, block_size: int = 512):
    """
    Memory-efficient Laplacian score.
    X: (n_samples x n_features) dense or sparse
    W: sparse affinity matrix (n_samples x n_samples)
    Computes in blocks to avoid holding (W @ X) for all genes at once.
    """
    from scipy.sparse import issparse, csr_matrix

    if not isinstance(W, csr_matrix):
        W = csr_matrix(W)
    n_samples = W.shape[0]

    if issparse(X):
        X_csr = X.tocsr()
    else:
        X_csr = csr_matrix(np.asarray(X, dtype=np.float32))

    D = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    D_sum = D.sum()
    # Precompute sum_i D_i X_{i,j} and sum_i D_i X_{i,j}^2
    # sum_D_X = X^T D
    sum_D_X = (X_csr.T).dot(D)  # (n_features,)
    sum_D_X2 = (X_csr.power(2).T).dot(D)

    D_prime = sum_D_X2 - (sum_D_X**2) / D_sum
    D_prime = np.maximum(D_prime, 1e-12)

    n_features = X_csr.shape[1]
    L_prime = np.empty(n_features, dtype=np.float32)

    for start in range(0, n_features, block_size):
        end = min(start + block_size, n_features)
        X_block = X_csr[:, start:end]
        WX_block = W.dot(X_block)
        # sum_i WX_ij * X_ij -> element-wise multiply then sum rows
        prod = WX_block.multiply(X_block)
        L_prime[start:end] = np.asarray(prod.sum(axis=0)).ravel() - (sum_D_X[start:end] ** 2) / D_sum

    score = 1.0 - (L_prime / D_prime)
    return score


def feature_ranking(score):
    """
    Rank features in ascending order according to their laplacian scores, the smaller the laplacian score is, the more
    important the feature is
    """
    idx = np.argsort(score, 0)
    return idx


def get_svg(adata, n_top, kneighbors=18):
    """
    Get the top n features according to the laplacian score
    """
    sim_graph = SimilarityGraph(adata, kneighbors=kneighbors)  # type: ignore
    lapScore = lap_score(adata.X, csr_matrix(sim_graph.neighbor_corr))
    top_feature = feature_ranking(lapScore)
    genelist = adata.var_names[top_feature[:n_top]]
    return genelist


def data_preprocess(adata, min_cells=3, top_hvg=None):
    """preprocessing adata"""
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    if top_hvg is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=top_hvg)
        adata = adata[:, adata.var.highly_variable]
    return adata


def refine_label(adata, radius=25, key="mclust"):
    """
    Refine the clustering results by majority voting
    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm["spatial"]
    distance = np.linalg.norm(position[:, np.newaxis] - position[np.newaxis, :], axis=2)
    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def clustering(adata, n_clusters, method="mclust", refine=False, dims=18, radius=25, seed=2025):
    """
    Clustering adata using the mclust algorithm
    """

    sc.tl.pca(adata, n_comps=dims)
    if method == "mclust":
        print("Clustering using mclust")
        adata = mclust_R(adata, used_obsm="X_pca", num_cluster=n_clusters, random_seed=seed)
        adata.obs["domain"] = adata.obs["mclust"]
    if method == "leiden":
        print("Clustering using leiden")
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=0.5)
        adata.obs["domain"] = adata.obs["leiden"]
    if refine:
        print("Refining the clustering results by majority voting")
        adata.obs["domain"] = refine_label(adata, radius=radius, key=method)


def iou_score(arr1, arr2):
    intersection = np.logical_and(arr1, arr2)
    union = np.logical_or(arr1, arr2)
    return np.sum(intersection) / np.sum(union)


def iou_rank(cluster, labels):
    cluster = cluster[:, np.newaxis]
    intersection = np.logical_and(cluster, labels).sum(axis=0)
    union = np.logical_or(cluster, labels).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union != 0, intersection / union, 0)

    ranked_indices = np.argsort(ious)[::-1]
    return ranked_indices
