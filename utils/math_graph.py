import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs


def scaled_laplacian(wa):
    """
    Normalized graph Laplacian function.
    :param wa: np.ndarray, [n_well, n_well], weighted adjacency matrix of G.
    :return: np.matrix, [n_well, n_well].
    """
    # d -> diagonal degree matrix
    n, d = np.shape(wa)[0], np.sum(wa, axis=1)
    # la -> normalized graph Laplacian
    la = -wa
    la[np.diag_indices_from(la)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                la[i, j] = la[i, j] / np.sqrt(d[i] * d[j])
    lambda_max = eigs(la, k=1, which='LR')[0][0].real
    return np.mat(2 * la / lambda_max - np.identity(n))


def cheb_poly_approx(la, ks, n):
    """
    Chebyshev polynomials approximation function.
    :param la: np.matrix, [n_well, n_well], graph Laplacian.
    :param ks: int, kernel size of spatial convolution.
    :param n: int, size of graph.
    :return: np.ndarray, [n_well, ks * n_well].
    """
    la0, la1 = np.mat(np.identity(n)), np.mat(np.copy(la))

    if ks > 1:
        la_list = [np.copy(la0), np.copy(la1)]
        for i in range(ks - 2):
            la_n = np.mat(2 * la * la1 - la0)
            la_list.append(np.copy(la_n))
            la0, la1 = np.mat(np.copy(la1)), np.mat(np.copy(la_n))
        return np.concatenate(la_list, axis=-1)
    elif ks == 1:
        return np.asarray(la0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received {ks}')


def first_approx(W, n):
    """
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    """
    A = W + np.identity(n)
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def weight_matrix(file_path, sigma2=0.6, epsilon=0.3, scaling=True):
    """
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix wa.
    :param epsilon: float, thresholds to control the sparsity of matrix wa.
    :param scaling: bool, whether applies numerical scaling on wa.
    :return: np.ndarray, [n_well, n_well].
    """
    try:
        wa = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}')

    # check whether wa is a 0/1 matrix.
    if set(np.unique(wa)) == {0, 1}:
        print('The input graph is a 0/1 matrix, set "scaling" to False.')
        scaling = False

    if scaling:
        n = wa.shape[0]
        wa = wa / 10000.  # change the scaling number if necessary
        wa2, wa_mask = wa * wa, np.ones([n, n]) - np.identity(n)
        return np.exp(-wa2 / sigma2) * (np.exp(-wa2 / sigma2) >= epsilon) * wa_mask
    else:
        return wa

