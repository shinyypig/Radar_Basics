# %%
import plotly.graph_objects as go
import numpy as np
import matplotlib.image as mpimg
import matplotlib
from scipy import signal
import tensorly as tl
from tensorly.decomposition import parafac
import pandas as pd
from scipy.linalg import null_space


def correlate_along_axis(Y, s, axis=0, mode="same", conjugate_template=False):
    """
    对多维数组 Y 的指定轴逐通道做 1D 相关（np.correlate），其余维度保持不变。
    - Y: ndarray，实数或复数。假设在 `axis` 上是时间/序列维。
    - s: 1D 模板序列（若传入 2D/3D，会 squeeze 到 1D；再不行就报错）。
    - axis: 相关所沿的轴（默认 0）。
    - mode: 'full' | 'same' | 'valid'（同 np.correlate）。
    - conjugate_template: 若为 True，使用 np.conj(s) 做互相关（复数常用）。
    返回：与 Y 同阶的 ndarray，`axis` 维长度按 mode 改变，其余维度不变。
    """
    Y = np.asarray(Y)
    s = np.asarray(s)

    # 把模板压成 1D
    s = np.squeeze(s)
    if s.ndim != 1:
        raise ValueError(f"s must be 1D after squeeze; got shape {s.shape}")

    # 复数互相关：x ⋆ s = correlate(x, conj(s))
    if conjugate_template:
        s_eff = np.conj(s)
    else:
        s_eff = s

    # 将目标轴移到最后，便于逐列处理
    Y_move = np.moveaxis(Y, axis, -1)  # (..., T)
    tail_len = Y_move.shape[-1]

    # 逐向量做相关
    def _corr_1d(vec):
        return signal.correlate(vec, s_eff, mode=mode)

    # apply_along_axis 会保留前面的维度结构
    out = np.apply_along_axis(_corr_1d, -1, Y_move)

    # 把轴移回原位
    out = np.moveaxis(out, -1, axis)
    return out


def tensor_mul(T: np.ndarray, X: np.ndarray, d: int) -> np.ndarray:
    """
    n-模乘（与给定的 MATLAB 实现等价），支持复数。
    给定张量 T 和矩阵 X，将在第 d 个维度上做乘法，并将该维度大小由 I_d 变为 J。

    约定（与原 MATLAB 代码一致）：
    - d 为 1-based 维度索引；
    - 形状匹配要求：X.shape == (I_d, J)，其中 I_d = T.shape[d-1]；
    - 计算用的是 X'（共轭转置），即复数情况下使用共轭。

    等价于张量 n-模乘：M = T ×_d (X^H)^T，其中 X^H 是 X 的共轭转置；
    与 MATLAB 代码中的 M = X' * T 的行为一致（' 为共轭转置）。

    参数
    ----
    T : np.ndarray
        待变换的张量（可为复数）
    X : np.ndarray
        矩阵，形状应为 (I_d, J)
    d : int
        1-based 的乘法维度索引

    返回
    ----
    M : np.ndarray
        结果张量，形状与 T 相同但第 d 个维度由 I_d 变为 J
    """
    if d < 1:
        raise ValueError("d must be a positive 1-based dimension index.")

    # 若需要，补齐尾部的 1 维，确保 d 不超过张量维数
    if d > T.ndim:
        pad = d - T.ndim
        T = T.reshape(T.shape + (1,) * pad)

    I_d = T.shape[d - 1]
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if X.shape[0] != I_d:
        raise ValueError(
            f"X.shape[0] ({X.shape[0]}) must equal size of T along axis d ({I_d})."
        )

    # tensordot：将 X^H（= X.conj().T，形状 (J, I_d)）与 T 在轴 (I_d, d-1) 上收缩
    # 结果轴顺序：先得到 (J, 其余轴按 T 中 d-1 轴移除后的顺序)
    M_tmp = np.tensordot(X.conj().T, T, axes=([1], [d - 1]))  # 形状：(J, ...)

    # 将前置的维度 J 移回到第 d-1 个位置（把 axis 0 移到 d-1）
    # 目标：结果与 T 同阶，且第 d 个维度由 I_d 变为 J
    M = np.moveaxis(M_tmp, 0, d - 1)
    return M


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def receive(t):
    kappa = 40
    return rect(t) * np.exp(1j * 2 * np.pi * kappa * t**2)


L = 1000
M = 100

w1 = 0.1
w2 = -0.2
v1 = 0.15
v2 = -0.25

t = np.linspace(-2, 2, L)

s = receive(t)
# use a boolean mask with bitwise & (and needs elementwise comparison)
mask = (t >= -1) & (t <= 1)
s = s[mask]
s = s / np.linalg.norm(s)
s = np.reshape(s, (-1, 1))

r1 = receive(t + 1)
a1 = np.exp(1j * 2 * np.pi * w1 * np.arange(M))
b1 = np.exp(1j * 2 * np.pi * v1 * np.arange(M))

r2 = receive(t - 1)
a2 = np.exp(1j * 2 * np.pi * w2 * np.arange(M))
b2 = np.exp(1j * 2 * np.pi * v2 * np.arange(M))

X = np.reshape(r1, [-1, 1, 1]) * np.reshape(a1, [1, -1, 1]) * np.reshape(
    b1, [1, 1, -1]
) + np.reshape(r2, [-1, 1, 1]) * np.reshape(a2, [1, -1, 1]) * np.reshape(b2, [1, 1, -1])

X += 1 * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))

# Y = correlate_along_axis(X, s, mode="same")
# Y = tensor_mul(Y, W, 2)
# Y = tensor_mul(Y, W, 3)

# Y = np.mean(np.abs(Y) ** 2, axis=0)
# Y = Y.squeeze()
# Y = Y / Y.max()
# mpimg.imsave("stap1.png", Y, dpi=300, cmap="viridis")

# %%

factors = parafac(X, rank=2)

R, A, B = factors.factors

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(R[:, 0]), mode="lines", name="r1"))
fig.add_trace(go.Scatter(y=np.real(R[:, 1]), mode="lines", name="r2"))
fig.update_layout(title="Estimated r")
fig.show()

df = pd.DataFrame({"x": t, "r1": np.real(R[:, 0]), "r2": np.real(R[:, 1])})
df.to_csv("cp_r.csv", index=False)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(A[:, 0]) / (2 * np.pi), mode="lines", name="a1"))
fig.add_trace(go.Scatter(y=np.real(A[:, 1]) / (2 * np.pi), mode="lines", name="a2"))
fig.update_layout(title="Estimated a (phase / 2pi)")
fig.show()

df = pd.DataFrame({"x": np.arange(M), "a1": np.real(A[:, 0]), "a2": np.real(A[:, 1])})
df.to_csv("cp_a.csv", index=False)

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(B[:, 0]) / (2 * np.pi), mode="lines", name="b1"))
fig.add_trace(go.Scatter(y=np.real(B[:, 1]) / (2 * np.pi), mode="lines", name="b2"))
fig.update_layout(title="Estimated b (phase / 2pi)")
fig.show()

df = pd.DataFrame({"x": np.arange(M), "b1": np.real(B[:, 0]), "b2": np.real(B[:, 1])})
df.to_csv("cp_b.csv", index=False)

# %%

V = tl.tenalg.khatri_rao([A, B])
N = null_space(V.T)

W = np.exp(
    1j * 2 * np.pi * np.linspace(-0.5, 0.5, 200)[:, None] * np.arange(M)[None, :]
)
W = W.T
W = np.kron(W, W)

# %%
p = 1 / np.linalg.norm(N.T @ W, axis=0) ** 2
p = p / np.max(p)
p = np.reshape(p, [200, 200])

fig = go.Figure()
fig.add_trace(go.Heatmap(z=p, name="CP-MUSIC spectrum"))
fig.update_layout(title="CP-MUSIC spectrum")
fig.show()

mpimg.imsave("cp_music.png", p, dpi=300, cmap="viridis")
