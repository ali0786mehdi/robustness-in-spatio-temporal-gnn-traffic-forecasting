"""
Graph construction module.
Builds adjacency matrices from sensor time-series data using
correlation-based Gaussian kernel similarity.
"""

import numpy as np
from scipy.sparse import coo_matrix
import os


def compute_correlation_adj(data, sigma=0.1, epsilon=0.3):
    """
    Compute adjacency matrix from sensor time-series correlation.

    1. Compute pairwise Pearson correlation between all sensor pairs.
    2. Apply Gaussian kernel: A[i,j] = exp(-((1 - corr)^2) / (2 * sigma^2))
    3. Threshold: set A[i,j] = 0 if below epsilon.
    4. Add self-loops.

    Args:
        data (np.ndarray): Sensor time-series data, shape (T, N) where
                           T = timesteps, N = number of sensors.
        sigma (float): Gaussian kernel bandwidth.
        epsilon (float): Sparsity threshold.

    Returns:
        np.ndarray: Adjacency matrix, shape (N, N).
    """
    num_sensors = data.shape[1]

    # Compute Pearson correlation matrix
    # Use only non-zero entries to avoid issues with missing data
    corr_matrix = np.corrcoef(data.T)  # (N, N)

    # Handle NaN correlations (e.g., constant sensor)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Apply Gaussian kernel
    dist = 1.0 - corr_matrix  # distance = 1 - correlation
    adj = np.exp(-(dist ** 2) / (2 * sigma ** 2))

    # Apply sparsity threshold
    adj[adj < epsilon] = 0.0

    # Set diagonal to 1 (self-loops)
    np.fill_diagonal(adj, 1.0)

    print(f"  Adjacency matrix: {num_sensors}x{num_sensors}")
    print(f"  Non-zero entries: {np.count_nonzero(adj)} / {num_sensors ** 2}")
    print(f"  Sparsity: {1 - np.count_nonzero(adj) / (num_sensors ** 2):.2%}")
    print(f"  Avg connections per node: {np.count_nonzero(adj).sum() / num_sensors:.1f}")

    return adj


def symmetric_normalize(adj):
    """
    Symmetric normalization: D^(-1/2) A D^(-1/2)
    Used for GCN-style convolution (STGCN).

    Args:
        adj (np.ndarray): Adjacency matrix, shape (N, N).

    Returns:
        np.ndarray: Normalized adjacency matrix.
    """
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def random_walk_normalize(adj):
    """
    Random walk normalization: D^(-1) A
    Used for DCRNN diffusion convolution.

    Args:
        adj (np.ndarray): Adjacency matrix, shape (N, N).

    Returns:
        np.ndarray: Row-normalized adjacency matrix.
    """
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv = np.power(degree, -1.0)
    d_inv[np.isinf(d_inv)] = 0.0
    D_inv = np.diag(d_inv)
    return D_inv @ adj


def compute_chebyshev_polynomials(adj_normalized, K):
    """
    Compute Chebyshev polynomials of the graph Laplacian up to order K.
    Used for STGCN's ChebNet convolution.

    L = I - D^(-1/2) A D^(-1/2)
    L_tilde = 2L / lambda_max - I  (scaled Laplacian)

    Args:
        adj_normalized (np.ndarray): Symmetrically normalized adjacency matrix.
        K (int): Maximum polynomial order.

    Returns:
        list[np.ndarray]: List of K+1 Chebyshev polynomial matrices.
    """
    N = adj_normalized.shape[0]
    I = np.eye(N)

    # Graph Laplacian
    L = I - adj_normalized

    # Compute largest eigenvalue for scaling
    try:
        from scipy.sparse.linalg import eigsh
        L_sparse = coo_matrix(L)
        lambda_max = eigsh(L_sparse, k=1, which='LM', return_eigenvectors=False)[0]
    except Exception:
        lambda_max = 2.0  # Default for normalized Laplacian

    # Scaled Laplacian
    L_tilde = (2.0 / lambda_max) * L - I

    # Chebyshev polynomials: T_0 = I, T_1 = L_tilde, T_k = 2*L_tilde*T_{k-1} - T_{k-2}
    cheb_polys = [I, L_tilde]
    for k in range(2, K + 1):
        cheb_polys.append(2 * L_tilde @ cheb_polys[-1] - cheb_polys[-2])

    return cheb_polys[:K + 1]


def compute_diffusion_matrices(adj, K=2):
    """
    Compute diffusion transition matrices for DCRNN.
    Bidirectional: forward (D_O^{-1} W) and backward (D_I^{-1} W^T).

    Args:
        adj (np.ndarray): Adjacency matrix, shape (N, N).
        K (int): Number of diffusion steps.

    Returns:
        list[np.ndarray]: List of 2K diffusion matrices
                          [P_f^1, ..., P_f^K, P_b^1, ..., P_b^K]
    """
    # Forward transition: D_out^{-1} W
    P_forward = random_walk_normalize(adj)

    # Backward transition: D_in^{-1} W^T
    P_backward = random_walk_normalize(adj.T)

    # Compute powers
    supports = []
    Pf_k = P_forward.copy()
    Pb_k = P_backward.copy()
    for k in range(K):
        if k == 0:
            supports.append(P_forward)
            supports.append(P_backward)
        else:
            Pf_k = Pf_k @ P_forward
            Pb_k = Pb_k @ P_backward
            supports.append(Pf_k)
            supports.append(Pb_k)

    return supports


def build_graph(data, sigma=0.1, epsilon=0.3, K_cheb=3, K_diff=2, ablation=None):
    """
    Build complete graph structures from TRAIN DATA ONLY to prevent spatial data leakage.
    
    This function computes the correlation-based adjacency matrix, 
    Chebyshev polynomials for STGCN, and diffusion supports for DCRNN.

    Args:
        data (np.ndarray): Raw training data, shape (T_train, N).
                           MUST NOT contain any validation or test data!
        sigma (float): Bandwidth for Gaussian kernel.
        epsilon (float): Threshold for sparsifying the graph.
        K_cheb (int): Order of Chebyshev polynomials.
        K_diff (int): Number of diffusion steps.
        ablation (str): Optional. 'random' or 'identity' to override graph.

    Returns:
        dict: Dictionary containing:
            - 'adj': Raw adjacency matrix
            - 'adj_sym': Symmetrically normalized adjacency
            - 'cheb_polys': Chebyshev polynomials for STGCN
            - 'diffusion_supports': Diffusion matrices for DCRNN
    """
    if ablation:
        print(f"Building ABLATION graph: {ablation.upper()}...")
        num_sensors = data.shape[1]
        if ablation == 'identity':
            adj = np.eye(num_sensors)
        elif ablation == 'random':
            rand_mat = np.random.rand(num_sensors, num_sensors)
            adj = (rand_mat + rand_mat.T) / 2  # Make symmetric
            np.fill_diagonal(adj, 1.0)         # Add self-loops
        else:
            raise ValueError("ablation must be 'identity' or 'random'")
    else:
        print("Building graph from sensor correlations...")
        # Step 1: Compute adjacency matrix
        adj = compute_correlation_adj(data, sigma=sigma, epsilon=epsilon)

    # Step 2: Symmetric normalization (for STGCN)
    adj_sym = symmetric_normalize(adj)

    # Step 3: Chebyshev polynomials (for STGCN)
    print("  Computing Chebyshev polynomials...")
    cheb_polys = compute_chebyshev_polynomials(adj_sym, K_cheb)

    # Step 4: Diffusion matrices (for DCRNN)
    print("  Computing diffusion matrices...")
    diffusion_supports = compute_diffusion_matrices(adj, K_diff)

    print("Graph construction complete.\n")

    return {
        "adj": adj,
        "adj_sym": adj_sym,
        "cheb_polys": cheb_polys,
        "diffusion_supports": diffusion_supports,
    }


def save_graph(graph_dict, save_dir, dataset_name):
    """Save graph data to disk."""
    save_path = os.path.join(save_dir, f"{dataset_name}_graph.npz")
    np.savez(
        save_path,
        adj=graph_dict["adj"],
        adj_sym=graph_dict["adj_sym"],
    )
    # Save Chebyshev polynomials separately (list of arrays)
    for i, poly in enumerate(graph_dict["cheb_polys"]):
        np.save(os.path.join(save_dir, f"{dataset_name}_cheb_{i}.npy"), poly)

    print(f"Graph saved to {save_path}")


def load_graph(save_dir, dataset_name, K_cheb=3):
    """Load graph data from disk."""
    save_path = os.path.join(save_dir, f"{dataset_name}_graph.npz")
    data = np.load(save_path)
    graph_dict = {
        "adj": data["adj"],
        "adj_sym": data["adj_sym"],
    }
    cheb_polys = []
    for i in range(K_cheb + 1):
        path = os.path.join(save_dir, f"{dataset_name}_cheb_{i}.npy")
        if os.path.exists(path):
            cheb_polys.append(np.load(path))
    graph_dict["cheb_polys"] = cheb_polys
    graph_dict["diffusion_supports"] = compute_diffusion_matrices(
        graph_dict["adj"], K=2
    )
    return graph_dict
