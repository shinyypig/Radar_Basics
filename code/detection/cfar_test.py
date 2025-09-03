# %%
import plotly.graph_objects as go
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.image as mpimg
import pandas as pd


def cfar(x, N, G, pfa=1e-6):
    """
    一维 CA-CFAR（功率域）:
    x : 1D ndarray，脉压后取模平方的序列（功率）
    N : 每侧总窗半宽（含保护+训练），总窗长 2N+1
    G : 每侧保护半宽，保护长 2G+1，要求 N > G

    返回:
        det : bool数组，长度与 x 相同，仅在有效位置给出检测结果
        thr : 阈值数组（gamma_i），无效位置为 NaN
        z   : 噪声估计（训练均值），无效位置为 NaN
        alpha : 本次使用的门限系数（由 pfa 与训练数计算）
    """
    x = np.asarray(x, dtype=float)
    L = x.size
    if N <= G:
        raise ValueError("必须满足 N > G。")
    if L < 2 * (N + G) + 1:
        raise ValueError("数据太短，无法形成滑窗。")

    # 目标虚警率（可按需修改）

    # 训练单元总数（左右各 N-G 个）
    Ntrain_side = N - G
    Ntrain = 2 * Ntrain_side

    # alpha = N (pfa^{-1/N} - 1) 的推广：这里 N 用 Ntrain
    alpha = Ntrain * (pfa ** (-1.0 / Ntrain) - 1.0)

    # 有效 CUT 索引范围
    i0 = N + G
    i1 = L - (N + G)  # 终止索引（不含）
    idx = np.arange(i0, i1)

    # 前缀和实现 O(1) 区间求和
    csum = np.concatenate(([0.0], np.cumsum(x)))

    # 左训练窗 [i-N, i-G-1] 之和：csum[i-G] - csum[i-N]
    left_sum = csum[idx - G] - csum[idx - N]
    # 右训练窗 [i+G+1, i+N] 之和：csum[i+N+1] - csum[i+G+1]
    right_sum = csum[idx + N + 1] - csum[idx + G + 1]

    z_i = (left_sum + right_sum) / Ntrain  # 训练均值
    thr_i = alpha * z_i  # 阈值
    det_i = x[idx] > thr_i  # 判决

    # 对齐到原长度
    det = np.zeros(L, dtype=bool)
    thr = np.full(L, np.nan)
    z = np.full(L, np.nan)
    det[idx] = det_i
    thr[idx] = thr_i
    z[idx] = z_i

    return thr


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def s1(t, f):
    return rect(t) * np.exp(1j * np.pi * f * t)


def s2(t, kappa):
    return rect(t) * np.exp(1j * np.pi * kappa * t**2)


def s3(t, kappa):
    return rect(t) * np.exp(1j * np.pi * kappa * t**3)


t = np.linspace(-5, 5, 1000)
r = s2(t, 50)

kappa = 100
# set random seed
np.random.seed(0)
x = r + 2 * (np.random.randn(*t.shape) + 1j * np.random.randn(*t.shape)) * (t + 10) / 10

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.real(x), mode="lines", name="Real Part"))
fig.show()

y = np.abs(signal.correlate(x, r, mode="same")) ** 2
thr1 = cfar(y, 20, 5, 1e-1)
thr2 = cfar(y, 20, 5, 1e-2)
thr3 = cfar(y, 20, 5, 1e-3)

thr1 = thr1 / np.max(y)
thr2 = thr2 / np.max(y)
thr3 = thr3 / np.max(y)
y = y / np.max(y)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="Real Part"))
fig.add_trace(go.Scatter(x=t, y=thr1, mode="lines", name="Threshold 1e-1"))
fig.add_trace(go.Scatter(x=t, y=thr2, mode="lines", name="Threshold 1e-2"))
fig.add_trace(go.Scatter(x=t, y=thr3, mode="lines", name="Threshold 1e-3"))
fig.show()


df = pd.DataFrame(
    {
        "t": t,
        "y": y,
        "t1": thr1,
        "t2": thr2,
        "t3": thr3,
    }
)
df.to_csv("cfar.csv", index=False)
