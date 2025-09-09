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

# set random seed
np.random.seed(1)
X += 0.5 * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))

Z = np.real(X)
Z = (Z - Z.min()) / (Z.max() - Z.min())
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha
mpimg.imsave("capon_1.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%

W = np.exp(
    1j * 2 * np.pi * np.linspace(-0.5, 0.5, 200)[:, None] * np.arange(M)[None, :]
)
W = W.T

Y = signal.correlate(X, s, mode="same")

Z_cfb = np.abs(Y @ W)

Z_cfb = Z_cfb / Z_cfb.max()

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z_cfb)[:, :, :3]  # 去掉 alpha
mpimg.imsave("capon_2.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%
Rxx = (X.conj().T @ X) / X.shape[0]
Rxx_inv = np.linalg.inv(Rxx)

W_capon = Rxx_inv @ W / np.mean(W.conj() * (Rxx_inv @ W), axis=0)

Y = signal.correlate(X, s, mode="same")

Z_capon = np.abs(Y @ W_capon)

Z_capon = Z_capon / Z_capon.max()

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z_capon)[:, :, :3]  # 去掉 alpha
mpimg.imsave("capon_3.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%
import pandas as pd

p = np.mean(Z_cfb**2, axis=0)
p = p / np.max(p)

# Y = 1 / (W.conj() @ Rxx_inv * W.conj())
# p2 = np.mean(np.abs(Y) ** 2, axis=1)
p2 = np.mean(Z_capon**2, axis=0)
p2 = p2 / np.max(p2)

fig = go.Figure(go.Scatter(y=p2, mode="lines"))
fig.add_trace(go.Scatter(y=p, mode="lines"))
fig.show()

df = pd.DataFrame({"w": np.linspace(-0.5, 0.5, 200), "p1": p[::-1], "p2": p2[::-1]})
df.to_csv("capon.csv", index=False)
