# %%
import plotly.graph_objects as go
import numpy as np
import matplotlib.image as mpimg
import matplotlib
from scipy import signal
from utils import *
import pandas as pd


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def receive(t):
    kappa = 40
    return rect(t) * np.exp(1j * 2 * np.pi * kappa * t**2)


L = 1000
M = 40

w1 = 0.1
w2 = -0.2

t = np.linspace(-2, 2, L)

s = receive(t)
# use a boolean mask with bitwise & (and needs elementwise comparison)
mask = (t >= -1) & (t <= 1)
s = s[mask]
s = s / np.linalg.norm(s)
s = np.reshape(s, (-1, 1))

r1 = receive(t + 1)
a1 = np.exp(1j * 2 * np.pi * w1 * np.arange(M))

r2 = receive(t - 1)
a2 = np.exp(1j * 2 * np.pi * w2 * np.arange(M))

R = np.vstack([r1, r2]).T
A = np.vstack([a1, a2]).T

X = R @ A.T

# set random seed
np.random.seed(1)
X += 0.5 * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))

Z = np.real(X)
Z = (Z - Z.min()) / (Z.max() - Z.min())
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("l1svd_1.png", Z[:, ::-1, :], dpi=300)

# %%
N = 200
W = np.exp(
    1j * 2 * np.pi * np.linspace(-0.5, 0.5, N)[:, None] * np.arange(X.shape[1])[None, :]
)
D = W.T

[U, S, V] = np.linalg.svd(X)
Us = U[:, :2]
X_ = Us.T @ X

# %%
import numpy as np
import cvxpy as cp


def solve_socp(
    X: np.ndarray, D: np.ndarray, lam: float, solver_order=("MOSEK", "ECOS", "SCS")
):
    """
    Solve:
        minimize     ep + lam * ev
        subject to   || X - S D^T ||_F <= ep
                     sum_i || S[:, i] ||_2  <= ev     (i.e., l2,1-范数 <= ev)
    where S is complex (L x I).
    """
    L, M = X.shape
    M2, I = D.shape
    assert M2 == M, "D must have M rows to match X's columns."

    # Decision variables
    S = cp.Variable((L, I), complex=True)
    ep = cp.Variable(nonneg=True)  # ε (residual bound)
    ev = cp.Variable(nonneg=True)  # ε' (regularizer bound)

    # Constraints
    cons = [
        # 使用 Frobenius 范数，等价于 ||vec(·)||_2，避免 vec 次序警告
        cp.norm(X - S @ D.T, "fro") <= ep,
        # l2,1 约束：列范数求和
        cp.sum(cp.norm(S, axis=0)) <= ev,
    ]

    # Objective
    obj = cp.Minimize(ep + lam * ev)
    prob = cp.Problem(obj, cons)

    # Pick a solver that is installed
    for s in solver_order:
        if s in cp.installed_solvers():
            try:
                prob.solve(solver=getattr(cp, s))
                break
            except Exception:
                continue
    else:
        prob.solve()

    return {
        "status": prob.status,
        "opt_val": prob.value,
        "S": S.value,  # complex (L x I)
        "ep": float(ep.value) if ep.value is not None else None,
        "ev": float(ev.value) if ev.value is not None else None,
        "solver": (
            prob.solver_stats.solver_name if prob.solver_stats is not None else None
        ),
    }


out = solve_socp(X, D, lam=3)
p = np.linalg.norm(out["S"], axis=0) ** 2
p = p / np.max(p)
p = p[::-1]

p_music, _ = music(X, N, 2)

p = np.log10(p) * 10
p_music = np.log10(p_music) * 10
go.Figure(
    [
        go.Scatter(
            x=np.linspace(-0.5, 0.5, N),
            y=p_music,
            mode="markers+lines",
            marker=dict(size=5),
        ),
        go.Scatter(
            x=np.linspace(-0.5, 0.5, N), y=p, mode="markers+lines", marker=dict(size=5)
        ),
    ]
).show()

# %%
go.Figure(
    go.Heatmap(
        z=np.real(out["S"]),
    )
).show()

# %%

Z = np.real(out["S"])
Z = (Z - Z.min()) / (Z.max() - Z.min())
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("l1svd_2.png", Z[:, ::-1, :], dpi=300)

# %%

df = pd.DataFrame({"x": np.linspace(-0.5, 0.5, N), "p1": p_music, "p2": p})
df.to_csv("l1svd.csv", index=False)
