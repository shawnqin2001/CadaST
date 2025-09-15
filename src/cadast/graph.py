import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


class SimilarityGraph:
    """
    Construct Similarity graph and implement HMRF in spatial transcriptomics
    """

    def __init__(
        self,
        adata,
        kneighbors,
        beta,
        alpha,
        theta,
        init_alpha,
        icm_iter: int = 3,
        max_iter: int = 3,
        n_components: int = 2,
        convergency_threshold: float = 1e-5,
        verbose: bool = False,
        memory_efficient: bool = True,
    ) -> None:
        self.verbose = verbose
        # Keep original matrix (sparse or dense) but enforce float32 without densifying
        X = adata.X
        if hasattr(X, "astype"):
            self.matrix = X.astype(np.float32)
        else:
            self.matrix = np.asarray(X, dtype=np.float32)
        self.cell_num = adata.shape[0]
        self.kneighbors = kneighbors + 1
        self.memory_efficient = memory_efficient
        # Get KNN indices once (avoid building large intermediate if memory_efficient)
        self._build_knn(adata.obsm["spatial"], self.kneighbors)
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.init_alpha = init_alpha
        self.icm_iter = icm_iter
        self.max_iter = max_iter
        self.n_components = n_components
        self.convergency_threshold = convergency_threshold
        # Build a lightweight adjacency (csr) from indices (no distances needed for correlation)
        self.graph = self._knn_indices_to_csr(self.cell_neighbors, self.cell_num)
        self.neighbor_corr = self._neighbor_init(self.alpha)
        if self.verbose:
            print(f"Initialized model with beta: {self.beta}, alpha: {alpha}, theta: {self.theta}")

    def _build_knn(self, coord: np.ndarray, kneighbors: int):
        if self.verbose:
            print("Constructing KNN")
        nbrs = NearestNeighbors(n_neighbors=kneighbors, algorithm="auto")
        nbrs.fit(coord)
        distances, indices = nbrs.kneighbors(coord, return_distance=True)
        self.cell_neighbors = indices.astype(np.int32)
        self.neighbor_distances = distances.astype(np.float32)

    @staticmethod
    def _knn_indices_to_csr(indices: np.ndarray, n_cells: int) -> csr_matrix:
        # Directed KNN graph
        rows = np.repeat(np.arange(n_cells, dtype=np.int32), indices.shape[1])
        cols = indices.ravel()
        data = np.ones_like(cols, dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    def _neighbor_init(self, alpha, n_comp=15) -> csr_matrix:
        """
        Initialize neighbor correlation matrix
        """
        if self.verbose:
            print("Initializing neighbor correlation matrix ")

        X = self.matrix
        if isspmatrix_csr(X):
            svd = TruncatedSVD(n_components=min(n_comp, X.shape[1] - 1))
            comps = svd.fit_transform(X)
        else:
            svd = TruncatedSVD(n_components=min(n_comp, X.shape[1] - 1))
            comps = svd.fit_transform(X)

        comps = comps.astype(np.float32, copy=False)

        comps -= comps.mean(axis=1, keepdims=True)
        row_norms = np.linalg.norm(comps, axis=1)
        row_norms[row_norms == 0] = 1e-5
        comps /= row_norms[:, None]

        idx = self.cell_neighbors
        n_cells, k = idx.shape
        rows = np.repeat(np.arange(n_cells, dtype=np.int32), k)
        cols = idx.ravel()
        block = 4096
        sims = np.empty(rows.shape[0], dtype=np.float32)
        write_ptr = 0
        for start in range(0, n_cells, block):
            end = min(start + block, n_cells)
            block_idx = idx[start:end]
            base = comps[start:end]
            dots = (base[:, None, :] * comps[block_idx]).sum(axis=2)
            bsz = (end - start) * k
            sims[write_ptr : write_ptr + bsz] = np.exp(dots.ravel())
            write_ptr += bsz

        neighbor_corr = csr_matrix((sims, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)
        neighbor_corr.setdiag(0)
        self._csr_row_normalize_inplace(neighbor_corr)
        neighbor_corr.data *= alpha
        neighbor_corr.setdiag(1.0)
        self._csr_row_normalize_inplace(neighbor_corr)
        return neighbor_corr

    def _update_adj_matrix(self, theta: float) -> None:
        """
        Scale edges whose labels differ by theta (in-place friendly).
        """
        nc = self.neighbor_corr
        coo = nc.tocoo()
        row = coo.row
        col = coo.col
        data = coo.data.copy()
        diff = self.labels[row] != self.labels[col]
        data[diff] *= theta
        adj = csr_matrix((data, (row, col)), shape=nc.shape)
        self._csr_row_normalize_inplace(adj)
        self.adj_matrix = adj

    @staticmethod
    def _csr_row_normalize_inplace(mat: csr_matrix):
        """
        In-place row normalization (avoids allocating new large arrays).
        """
        row_sums = np.array(mat.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        inv = (1.0 / row_sums).astype(mat.data.dtype)

        row_idx = np.repeat(np.arange(mat.shape[0], dtype=np.int32), np.diff(mat.indptr))
        mat.data *= inv[row_idx]
        return mat

    def fit(self, gene_idx: int) -> None:
        """
        Implement HMRF using ICM-EM
        """
        self.exp = self.matrix[:, gene_idx]
        self.exp = self.exp.toarray().ravel() if hasattr(self.exp, "toarray") else self.exp
        self._initialize_labels()
        self._update_adj_matrix(self.theta)
        self._run_icmem()

    def _initialize_labels(self) -> None:
        """
        Initialize label with smoothed expression matrix
        """
        neighbor_corr = self.neighbor_corr.copy()
        neighbor_corr = neighbor_corr / self.alpha * self.init_alpha
        neighbor_corr.setdiag(1)
        smoothed_exp = neighbor_corr.dot(self.exp)
        gmm = GaussianMixture(n_components=self.n_components).fit(smoothed_exp.reshape(-1, 1))
        means, covs = gmm.means_.ravel(), gmm.covariances_.ravel()  # type: ignore
        self.cls_para = np.column_stack((means, covs))
        self.labels = gmm.predict(smoothed_exp.reshape(-1, 1))
        if self.n_components:
            self._label_resort()

    def _impute(self) -> csr_matrix:
        """
        Impute the expression by considering neighbor cells
        """
        return self.adj_matrix.dot(self.exp)

    def _label_resort(self) -> None:
        """
        Set the label with the highest mean as 1
        """
        means = self.cls_para[:, 0]
        sorted_indices = np.argsort(means)
        label_map = np.zeros(self.n_components, dtype=int)
        label_map[sorted_indices] = np.arange(self.n_components)
        self.labels = label_map[self.labels]

    def _run_icmem(
        self,
        convergency_threshold: float = 1e-5,
    ) -> None:
        """
        Run ICM-EM algorithm to update gene panel's labels and integrate neighbor spots expression
        """
        beta = self.beta
        theta = self.theta
        icm_iter = self.icm_iter
        max_iter = self.max_iter
        sqrt2pi = np.sqrt(2 * np.pi)
        cell_num = self.cell_num
        temp = 1  # TODO add melting mechanism
        iteration = 0
        converged = False
        while iteration < max_iter and not converged:
            # ICM step
            changed = 0
            for _ in range(icm_iter):
                indices = np.arange(cell_num)
                new_labels = np.random.randint(0, self.n_components, size=cell_num)
                delta_energies = self._delta_energies(indices, new_labels, beta)
                negative_indices = delta_energies < 0
                self.labels[indices[negative_indices]] = new_labels[negative_indices]
                changed += np.sum(negative_indices)

                # Metropolis-Hastings
                # non_negative_indices = np.logical_not(negative_indices)
                # probabilities = np.exp(-delta_energies[non_negative_indices] / temp)
                # probabilities[probabilities == 0] = 1e-5
                # samples = np.random.uniform(0, 1, size=probabilities.shape)
                # update = samples < probabilities
                # self.labels[indices[non_negative_indices][update]] = new_labels[non_negative_indices][update]
                # changed += np.sum(update)

                if changed == 0:
                    break

            # EM step initialization
            means, vars = self.cls_para.T
            vars[np.isclose(vars, 0)] = 1e-5
            squared_diff = (self.exp[:, None] - means) ** 2

            # E step
            clusterProb = np.exp(-0.5 * squared_diff / vars) / (sqrt2pi * np.sqrt(vars))
            clusterProb[np.isclose(clusterProb, 0)] = 1e-5
            clusterProb = clusterProb / clusterProb.sum(axis=1)[:, None]

            # M Step
            weights = clusterProb / clusterProb.sum(axis=0)
            means = np.sum(self.exp[:, None] * weights, axis=0)
            vars = np.sum(weights * squared_diff, axis=0) / weights.sum(axis=0)
            vars[np.isclose(vars, 0)] = 1e-5

            new_para = np.column_stack([means, vars])
            para_change = np.max(np.abs(new_para - self.cls_para))
            if para_change < convergency_threshold:
                converged = True
            self.cls_para = new_para
            # Update expression matrix
            if changed > 0:
                self._update_adj_matrix(theta)
            self.exp = self._impute()
            iteration += 1
        return

    def _delta_energies(self, indices, new_labels, beta) -> np.ndarray:
        """
        Calculate the energy difference between the current and proposed labels.
        """
        current_labels = self.labels[indices]  # Get current labels for these indices

        # Get parameters for current and new labels
        current_means = self.cls_para[current_labels, 0]
        current_vars = self.cls_para[current_labels, 1]
        new_means = self.cls_para[new_labels, 0]
        new_vars = self.cls_para[new_labels, 1]

        sqrt_2_pi_current_vars = np.sqrt(2 * np.pi * current_vars)
        sqrt_2_pi_new_vars = np.sqrt(2 * np.pi * new_vars)

        # Likelihood energy difference
        delta_energy_consts = (
            np.log(sqrt_2_pi_new_vars / sqrt_2_pi_current_vars)
            + ((self.exp[indices] - new_means) ** 2 / (2 * new_vars))
            - ((self.exp[indices] - current_means) ** 2 / (2 * current_vars))
        )

        # Spatial energy difference
        neighbor_labels = self.labels[self.cell_neighbors[indices]] 

        # Calculate neighbor interaction differences
        current_neighbor_diff = np.sum(current_labels[:, np.newaxis] != neighbor_labels, axis=1)
        new_neighbor_diff = np.sum(new_labels[:, np.newaxis] != neighbor_labels, axis=1)

        delta_energy_neighbors = beta * 2 * (new_neighbor_diff - current_neighbor_diff) / self.kneighbors

        return delta_energy_consts + delta_energy_neighbors

    @staticmethod
    def _difference(x, y):
        return np.abs(np.subtract(x, y.T))
