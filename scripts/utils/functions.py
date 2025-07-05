def read_spatial_expression(
    file,
    sep="\s+",
    num_exp_genes=0.01,
    num_exp_spots=0.01,
    min_expression=1,
    drop=False,
):
    """
    Read raw data and returns pandas data frame of spatial gene express
    and numpy ndarray for single cell location coordinates;


    :param file: csv file for spatial gene expression;
    :rtype: coord (spatial coordinates) shape (n, 2); data: shape (n, m);
    """
    counts = pd.read_csv(file, sep=sep, index_col=0)
    print("raw data dim: {}".format(counts.shape))

    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    min_genes_spot_exp = round((counts != 0).sum(axis=1).quantile(num_exp_genes))
    print(
        "Number of expressed genes a spot must have to be kept "
        "({}% of total expressed genes) {}".format(num_exp_genes, min_genes_spot_exp)
    )

    mark_points = np.where((counts != 0).sum(axis=1) < min_genes_spot_exp)[0]
    print("Marked {} spots".format(len(mark_points)))

    if len(mark_points) > 0:
        noiseInd = [counts.shape[0] - 1 - i for i in range(len(mark_points))]
        if drop == False:
            temp = [val.split("x") for val in counts.index.values]
            coord = np.array([[float(a[0]), float(a[1])] for a in temp])

            similar_points = np.argsort(cdist(coord[mark_points, :], coord), axis=1)[
                :, 1
            ]
            for i, j in zip(mark_points, similar_points):
                counts.iloc[i, :] = counts.iloc[j, :]

            mark_counts = counts.iloc[mark_points, :]
            dropped_counts = counts.drop(counts.index[mark_points])
            counts = pd.concat([dropped_counts, mark_counts])

        else:
            counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    else:
        counts = counts
        noiseInd = []

    # Spots are columns and genes are rows
    counts = counts.transpose()
    # Remove noisy genes
    min_features_gene = round(len(counts.columns) * num_exp_spots)
    print(
        "Removing genes that are expressed in less than {} "
        "spots with a count of at least {}".format(min_features_gene, min_expression)
    )
    counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
    print("Dropped {} genes".format(num_genes - len(counts.index)))

    temp = [val.split("x") for val in counts.columns.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])

    data = counts.transpose()

    return coord, data, noiseInd


def plot_function(adata):
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    ax = sc.pl.scatter(
        adata,
        alpha=1,
        x="imagerow",
        y="imagecol",
        color="clusters",
        title="H",
        show=False,
        size=500000 / adata.shape[0],
        color_map="bwr",
    )
    ax.set_aspect("equal", "box")


def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    id_tmp1 = np.asarray(
        [not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool
    )
    id_tmp2 = np.asarray(
        [not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool
    )
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)
