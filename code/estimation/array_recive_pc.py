# %%
import plotly.graph_objects as go
import numpy as np
import matplotlib.image as mpimg
import matplotlib
from scipy import signal


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def receive(t):
    kappa = 40
    return rect(t) * np.exp(1j * 2 * np.pi * kappa * t**2)


L = 1000
M = 20

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

Z = np.real(X)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("est_1.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%
# Convolve each column of X with the pulse s
# Y = np.array([np.convolve(X[:, i], s, mode="same") for i in range(X.shape[1])]).T

Y = signal.correlate(X, s, mode="same")

Z = np.real(Y)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("est_2.png", Z[:, ::-1, :], dpi=300, cmap="viridis")


fig = go.Figure(go.Scatter(y=np.real(Y[:, 0]), mode="lines"))
fig.show()

# %%
W = np.exp(
    1j * 2 * np.pi * np.linspace(-0.5, 0.5, 200)[:, None] * np.arange(M)[None, :]
)
Y2 = X @ W.T
Z = np.real(Y2)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("est_3.png", Z[:, ::-1, :], dpi=300, cmap="viridis")


# %%
Y2 = Y @ W.T
Z = np.abs(Y2)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("est_4.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%
import pandas as pd

p = np.mean(np.abs(Y2) ** 2, axis=0)
p = p / np.max(p)
fig = go.Figure(go.Scatter(y=p, mode="lines"))
fig.show()

df = pd.DataFrame({"w": np.linspace(-0.5, 0.5, 200), "p": p[::-1]})
df.to_csv("cbf.csv", index=False)
