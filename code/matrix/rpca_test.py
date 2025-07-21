# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord="fro")

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1e-7 * self.frobenius_norm(self.D)

        # this loop implements the principal component pursuit (PCP) algorithm
        # located in the table on page 29 of https://arxiv.org/pdf/0912.3599.pdf
        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv
            )  # this line implements step 3
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda
            )  # this line implements step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)  # this line implements step 5
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print("iteration: {0}, error: {1}".format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print("ymin: {0}, ymax: {1}".format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], "r")
            plt.plot(self.L[n, :], "b")
            if not axis_on:
                plt.axis("off")


img = cv2.imread("img/matrix/cat_resized.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64) / 255.0  # 归一化
# add 椒盐噪声
noise = np.random.choice([0, 1], size=img.shape, p=[0.99, 0.01])
img[noise == 1] = np.random.choice([0, 1], size=np.sum(noise), p=[0.5, 0.5])

rpca = R_pca(D=img, lmbda=0.2)
L, S = rpca.fit(tol=1e-7, max_iter=1000, iter_print=100)


# %%
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(L, cmap="gray")
plt.title("Low-rank component")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(S, cmap="gray")
plt.title("Sparse component")
plt.axis("off")
plt.tight_layout()
plt.show()

# %%
S = (S - np.min(S)) / (np.max(S) - np.min(S))  # normalize sparse component
# save all images
cv2.imwrite("img/matrix/cat_low_rank.jpg", (L * 255).astype(np.uint8))
cv2.imwrite("img/matrix/cat_sparse.jpg", (S * 255).astype(np.uint8))
cv2.imwrite("img/matrix/cat_noisy.jpg", (img * 255).astype(np.uint8))
