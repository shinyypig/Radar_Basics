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

w1 = 0.05
w2 = -0.1

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
X += 0.1 * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))

Z = np.real(X)
Z = (Z - Z.min()) / (Z.max() - Z.min())
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("esprit_1.png", Z[:, ::-1, :], dpi=300)

# %%
[U, S, V] = np.linalg.svd(X)
Us = U[:, :2]
Vs = V.T[:, :2]

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(Us[:, 0]), mode="lines", name="U1"))
fig.add_trace(go.Scatter(y=np.real(Us[:, 1]), mode="lines", name="U2"))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(Vs[:, 0]), mode="lines", name="V1"))
fig.add_trace(go.Scatter(y=np.real(Vs[:, 1]), mode="lines", name="V2"))
fig.show()

df = pd.DataFrame({"x": np.arange(M), "V1": np.real(Vs[:, 0]), "V2": np.real(Vs[:, 1])})
df.to_csv("esprit_modes.csv", index=False)

df = pd.DataFrame(
    {"x": np.arange(R.shape[0]), "U1": np.real(Us[:, 0]), "U2": np.real(Us[:, 1])}
)
df.to_csv("esprit_responses.csv", index=False)

# %%
V1 = Vs[:-1, :]
V2 = Vs[1:, :]
Phi = np.linalg.pinv(V1) @ V2
[D, T] = np.linalg.eig(Phi)

A = Vs @ T
R = Us @ np.diag(S[:2]) @ np.linalg.pinv(T.T)

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(R[:, 0]), mode="lines", name="U1"))
fig.add_trace(go.Scatter(y=np.real(R[:, 1]), mode="lines", name="U2"))
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.real(A[:, 0]), mode="lines", name="V1"))
fig.add_trace(go.Scatter(y=np.real(A[:, 1]), mode="lines", name="V2"))
fig.show()

df = pd.DataFrame(
    {"x": np.arange(R.shape[0]), "R1": np.real(R[:, 0]), "R2": np.real(R[:, 1])}
)
df.to_csv("esprit_responses_refined.csv", index=False)

df = pd.DataFrame(
    {"x": np.arange(A.shape[0]), "A1": np.real(A[:, 0]), "A2": np.real(A[:, 1])}
)
df.to_csv("esprit_modes_refined.csv", index=False)

# %%
print("Estimated frequencies:")
print(np.angle(D) / (2 * np.pi))

# %%
p_music, X_music = cbf(X, 800)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.linspace(-0.5, 0.5, 800),
        y=p_music,
        mode="lines",
        line=dict(color="blue"),
    )
)
fig.show()

df = pd.DataFrame({"x": np.linspace(-0.5, 0.5, 800), "y": p_music})
df.to_csv("cbf_spectrum_ref.csv", index=False)
