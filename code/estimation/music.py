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
X += 2 * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))

Z = np.real(X)
Z = (Z - Z.min()) / (Z.max() - Z.min())
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("music_1.png", Z[:, ::-1, :], dpi=300)
# %%
# [U, S, V] = np.linalg.svd(X)

# Us = U[:, :2]
# Vs = V.T[:, :2]

# Xs = Us @ Vs.T
N = 800
p_cbf, X_cbf = cbf(X, N)
p_capon, X_capon = capon(X, N)
p_music, X_music = music(X, N, 2)

Z = np.real(X_music)
Z = (Z - Z.min()) / (Z.max() - Z.min())
# cmap = matplotlib.colormaps["viridis"]
# Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("music_2.png", Z, dpi=300, cmap="viridis")

go.Figure(
    data=[
        go.Scatter(
            x=np.linspace(-0.5, 0.5, N),
            y=p_cbf,
            mode="lines",
            line=dict(color="red"),
        ),
        go.Scatter(
            x=np.linspace(-0.5, 0.5, N),
            y=p_capon,
            mode="lines",
            line=dict(color="green"),
        ),
        go.Scatter(
            x=np.linspace(-0.5, 0.5, N),
            y=p_music,
            mode="lines",
            line=dict(color="blue"),
        ),
    ],
    layout=go.Layout(
        title="MUSIC Spectrum",
        xaxis=dict(title="Frequency"),
        yaxis=dict(title="Normalized Power"),
    ),
).show()

df = pd.DataFrame(
    {
        "f": np.linspace(-0.5, 0.5, N),
        "p1": p_cbf,
        "p2": p_capon,
        "p3": p_music,
    }
)
df.to_csv("music_spectrum.csv", index=False)

# %%
Y = signal.correlate(X_cbf, s, mode="same")
Z = np.abs(Y)
Z = Z / np.max(Z)
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("music_3.png", Z[:, ::-1, :], dpi=300)

Y = signal.correlate(X_capon, s, mode="same")
Z = np.abs(Y)
Z = Z / np.max(Z)
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("music_4.png", Z[:, ::-1, :], dpi=300)

# %%

p_music, X_music = music(X, N, 2)
_, X_music_ = capon(X_music, N, reg=1e-5)
Y = signal.correlate(X_music_, s, mode="same")
Z = np.abs(Y)
Z = Z / np.max(Z)
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("music_5.png", Z[:, ::-1, :], dpi=300)
