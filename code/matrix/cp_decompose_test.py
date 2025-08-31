# %%
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd


def tensor_mul(T: np.ndarray, X: np.ndarray, d: int) -> np.ndarray:
    """
    TENSOR_MUL: 张量与矩阵在第 d 个维度（1-based）相乘。
    - T: ndarray, 形状 (n1, n2, ..., nD)
    - X: ndarray, 形状 (n_d, m)，要求 X.shape[0] == T 在第 d 个维度的长度
    - d: int, 乘法维度（1-based，与 MATLAB 保持一致）

    返回:
    - M: ndarray，形状与 T 相同，只是第 d 个维度从 n_d 变为 m
    """
    if d < 1:
        raise ValueError("d 必须是从 1 开始的正整数（1-based）。")

    T = np.asarray(T)
    X = np.asarray(X)

    # 若 d 超过当前维度数，则按 MATLAB 行为在尾部补 1 维
    while T.ndim < d:
        T = T.reshape(T.shape + (1,))

    D = T.ndim
    d0 = d - 1  # 转为 0-based

    # 校验维度匹配
    if X.shape[0] != T.shape[d0]:
        raise ValueError(
            f"维度不匹配: X.shape[0]={X.shape[0]} 必须等于 T.shape[{d0}]={T.shape[d0]}"
        )

    # 置换：把第 d0 维移到最前 [d, 1..d-1, d+1..D]（0-based）
    axes = [d0] + [i for i in range(D) if i != d0]
    T_perm = np.transpose(T, axes)  # 形状: (n_d, ...)

    # 按 MATLAB：T = reshape(T, Xshape(1), [])
    T_mat = T_perm.reshape(T.shape[d0], -1)  # (n_d, prod(其他维度))

    # M = X' * T  => (m, n_d) @ (n_d, N) = (m, N)
    M_mat = X.T @ T_mat  # 形状 (m, N)

    # 恢复回张量：第一维变成 m
    Mshape = list(T_perm.shape)
    Mshape[0] = X.shape[1]  # m
    M_perm = M_mat.reshape(Mshape)

    # 逆置换：按 MATLAB [2:d, 1, d+1:D]（转 0-based：range(1,d) + [0] + range(d,D)）
    back_perm = list(range(1, d)) + [0] + list(range(d, D))
    M = np.transpose(M_perm, back_perm)
    return M


# 读取图像
img = Image.open("img/matrix/cat.jpg")
# resize to 600 * 400
img = img.resize((300, 200))

img = np.array(img)

a = np.random.randn(200, 1)
b = np.random.randn(300, 1)
c = np.random.randn(3, 1)

R = 200

A = []
B = []
C = []
L = []

X = img.copy().reshape(200, 300, 3)

X_low = 0
lam_ = 0

for r in range(R):
    for t in range(100):
        a = tensor_mul(tensor_mul(X, b, 2), c, 3).reshape(200, 1)
        b = tensor_mul(tensor_mul(X, a, 1), c, 3).reshape(300, 1)
        c = tensor_mul(tensor_mul(X, a, 1), b, 2).reshape(3, 1)

        lam = np.linalg.norm(a)

        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        c = c / np.linalg.norm(c)

        if lam - lam_ < 1e-6:
            A.append(a)
            B.append(b)
            C.append(c)
            L.append(lam)
            break
        lam_ = lam

    # create the rank-one tensor and sub it from X
    X_ = a.reshape(200, 1, 1) * b.reshape(1, 300, 1) * c.reshape(1, 1, 3) * lam
    X_low = X_low + X_
    X = X - X_

    print(np.linalg.norm(X))

# %%
# imshow X_low
fig = go.Figure()
fig.add_trace(go.Image(z=X_low))
fig.show()

# %%
# 将L从大到小排列，A，B，C也同步
L = np.array(L)
A = np.array(A)
B = np.array(B)
C = np.array(C)

indices = np.argsort(L)[::-1]
L = L[indices]
A = A[indices]
B = B[indices]
C = C[indices]

# %%
df = pd.DataFrame({"x": range(1, len(L) + 1), "lambda": L})
df.to_csv("code/matrix/cp_decompose_result.csv", index=False)

# %%
# 分别保存 10, 20, 50 对应的彩色图像，以及完整重构 X_low
ranks = [10, 20, 50]
for k in ranks:
    k = min(k, L.shape[0])
    X_save = np.zeros_like(X_low, dtype=float)
    for i in range(k):
        X_ = (
            A[i].reshape(200, 1, 1)
            * B[i].reshape(1, 300, 1)
            * C[i].reshape(1, 1, 3)
            * L[i]
        )
        X_save = X_save + X_

    X_save_clipped = np.clip(X_save, 0, 255).astype(np.uint8)
    Image.fromarray(X_save_clipped).save(f"code/matrix/cat_recon_{k}.png")

# 保存最终累计的低秩重构（如果需要）
# X_low_clipped = np.clip(X_low, 0, 255).astype(np.uint8)
# Image.fromarray(X_low_clipped).save("code/matrix/cat_recon_full.png")
