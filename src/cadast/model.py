import numpy as np
from anndata import AnnData
from joblib import Parallel, delayed
from tqdm import tqdm

from .graph import SimilarityGraph
from .utils import feature_ranking, lap_score


class CadaST:
    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    Accerlate the process by using joblib multi-process
    """

    def __init__(
        self,
        adata: AnnData,
        kneighbors: int,
        beta: int = 10,
        alpha: float = 0.6,
        theta: float = 0.2,
        init_alpha: float = 6,
        icm_iter: int = 1,
        max_iter: int = 3,
        n_components: int = 2,
        n_top: int | None = None,
        n_jobs: int = 16,
        verbose: bool = True,
    ):
        self.adata = adata
        self.kneighbors = kneighbors
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.init_alpha = init_alpha
        self.max_iter = max_iter
        self.icm_iter = icm_iter
        self.n_components = n_components
        self.n_top = n_top
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.gene_list = self.adata.var_names
        self.graph = None

    def construct_graph(self) -> None:
        """
        Construct gene graph
        """
        graph = SimilarityGraph(
            adata=self.adata,
            kneighbors=self.kneighbors,
            beta=self.beta,
            alpha=self.alpha,
            theta=self.theta,
            init_alpha=self.init_alpha,
            icm_iter=self.icm_iter,
            max_iter=self.max_iter,
            n_components=self.n_components,
            verbose=self.verbose,
        )
        self.graph = graph

    def filter_genes(self, n_top=2000) -> None:
        """
        Filter genes with top SVG features
        """
        if self.graph is None:
            self.construct_graph()
        n_top = n_top if self.n_top is None else self.n_top
        if n_top is not None:
            if self.verbose:
                print(f"Filtering genes with top {n_top} SVG features")
            self.lapScore = lap_score(self.adata.X, self.graph.neighbor_corr)  # type: ignore
            self.feature_rank = feature_ranking(self.lapScore)
            self.feature_selected = self.feature_rank[:n_top]
            self.gene_list = self.gene_list[self.feature_selected]
            self.adata = self.adata[:, self.gene_list].copy()  # type: ignore

    def fit(self) -> AnnData:
        """
        Fit the CadaST model
        """
        if self.graph is None:
            self.construct_graph()
        if self.n_top is not None:
            self.feature_selected = np.arange(self.adata.shape[1])
            self.filter_genes()
        print("Start CadaST model fitting")
        n_cells = self.adata.n_obs
        n_genes = len(self.feature_selected)
        imputed_exp = np.empty((n_cells, n_genes), dtype=np.float32)
        labels = np.empty((n_cells, n_genes), dtype=np.int8)
        if self.n_jobs == 1:
            imputed_exp, labels = [], []
            for feature_idx in tqdm(self.feature_selected):
                self.graph.fit(
                    gene_idx=feature_idx,
                )
                imputed_exp.append(self.graph.exp)
                labels.append(self.graph.labels)
            self.adata.X = np.array(imputed_exp).T
            self.adata.layers["labels"] = np.array(labels).T
            return self.adata
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_gene)(
                col,
                self.graph,
                feature_idx,
            )
            for col, feature_idx in enumerate(tqdm(self.feature_selected))
        )
        for col, exp_vec, lab_vec in results:
            imputed_exp[:, col] = exp_vec.astype(np.float32, copy=False)
            labels[:, col] = lab_vec
        self.adata.X = imputed_exp
        self.adata.layers["labels"] = labels
        return self.adata

    @staticmethod
    def _process_gene(col_idx, model, feature_idx) -> tuple:
        model.fit(
            gene_idx=feature_idx,
        )
        return col_idx, model.exp, model.labels
