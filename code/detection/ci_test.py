# %%
import plotly.graph_objects as go
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.image as mpimg
import pandas as pd


def linear_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """复数一维线性卷积，返回长度 len(x)+len(h)-1"""
    x = np.asarray(x, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    y = np.zeros(x.size + h.size - 1, dtype=np.complex128)
    for n in range(y.size):
        # y[n] = sum_k x[k] * h[n-k]
        kmin = max(0, n - (h.size - 1))
        kmax = min(n, x.size - 1)
        s = 0.0 + 0.0j
        for k in range(kmin, kmax + 1):
            s += x[k] * h[n - k]
        y[n] = s
    return y


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def s(t, kappa):
    return rect(t) * np.exp(1j * np.pi * kappa * t**2)


t = np.linspace(-2, 2, 2000)

kappa = 100

X = []
for _ in range(10):
    X.append(
        s(t, kappa) + 4 * (np.random.randn(*t.shape) + 1j * np.random.randn(*t.shape))
    )

r = s(t, kappa)
r = r / np.linalg.norm(r)

Y = []
for x in X:
    y = np.convolve(x, np.conj(r), mode="same")
    Y.append(y)
y1 = np.mean(Y, axis=0)
y2 = np.sqrt(np.mean(np.abs(Y) ** 2, axis=0))

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.abs(Y[0]), mode="lines", name="Compressed Signal"))
fig.add_trace(go.Scatter(x=t, y=np.abs(y1), mode="lines", name="Compressed Signal"))
fig.add_trace(go.Scatter(x=t, y=y2, mode="lines", name="Compressed Signal"))
fig.show()

df = pd.DataFrame({"t": t, "y1": np.abs(Y[0]), "y2": np.abs(y1), "y3": y2})
df.to_csv("ci_eg.csv", index=False)


# %%
X = []
for _ in range(100):
    X.append(
        s(t, kappa) + 4 * (np.random.randn(*t.shape) + 1j * np.random.randn(*t.shape))
    )

r = s(t, kappa)
r = r / np.linalg.norm(r)

Y = []
for x in X:
    y = np.convolve(x, np.conj(r), mode="same")
    Y.append(y)
y1 = np.mean(Y, axis=0)
y2 = np.sqrt(np.mean(np.abs(Y) ** 2, axis=0))

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.abs(Y[0]), mode="lines", name="Compressed Signal"))
fig.add_trace(go.Scatter(x=t, y=np.abs(y1), mode="lines", name="Compressed Signal"))
fig.add_trace(go.Scatter(x=t, y=y2, mode="lines", name="Compressed Signal"))
fig.show()

df = pd.DataFrame({"t": t, "y1": np.abs(Y[0]), "y2": np.abs(y1), "y3": y2})
df.to_csv("ci_eg2.csv", index=False)
