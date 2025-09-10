import numpy as np
from scipy import signal
from numpy.linalg import svd, norm


def cbf(X, N):
    W = np.exp(
        1j
        * 2
        * np.pi
        * np.linspace(-0.5, 0.5, N)[:, None]
        * np.arange(X.shape[1])[None, :]
    )
    W = W.T

    Z = X @ W
    p = np.mean(np.abs(Z) ** 2, axis=0)
    p = p / np.max(p)

    return p, Z


def capon(X, N, reg=0):
    W = np.exp(
        1j
        * 2
        * np.pi
        * np.linspace(-0.5, 0.5, N)[:, None]
        * np.arange(X.shape[1])[None, :]
    )
    W = W.T

    Rxx = (X.conj().T @ X) / X.shape[0] + reg * np.eye(X.shape[1])
    Rxx_inv = np.linalg.inv(Rxx)

    W_capon = Rxx_inv @ W / np.mean(W.conj() * (Rxx_inv @ W), axis=0)

    Z = X @ W_capon
    p = np.mean(np.abs(Z) ** 2, axis=0)
    p = p / np.max(p)

    return p, Z


def music(X, N, d):
    U, _, V = svd(X)
    Us = U[:, :d]
    Vs = V.T[:, :d]
    Vn = V.T[:, d:]

    W = np.exp(
        1j
        * 2
        * np.pi
        * np.linspace(-0.5, 0.5, N)[:, None]
        * np.arange(X.shape[1])[None, :]
    )
    W = W.T
    print(W.shape, Vn.shape)
    p = 1 / norm(Vn.T @ W, axis=0) ** 2
    p = p / np.max(p)

    Z = Us @ Vs.T

    return p, Z
