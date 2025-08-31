# %%
import plotly.graph_objects as go
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.image as mpimg
import pandas as pd


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def s(t, kappa):
    return rect(t) * np.exp(1j * np.pi * kappa * t**2)


t = np.linspace(0, 5, 2000)

kappa = 100
x1 = s(t - 2, kappa) * np.exp(1j * np.random.uniform(-0.1, 0.1))
x2 = s(t - 2.2, kappa) * np.exp(1j * np.random.uniform(-0.1, 0.1))
x3 = s(t - 3.5, kappa) * np.exp(1j * np.random.uniform(-0.1, 0.1))

x = x1 + x2 + x3
x += 0.2 * np.random.normal(size=x.shape)

r = s(np.linspace(-0.5, 0.5, 400), kappa)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.real(x), mode="lines", name="Signal"))
fig.show()

# %%
# pulse compression

y = signal.convolve(x, np.conj(r), mode="same")
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.abs(y), mode="lines", name="Compressed Signal"))
fig.show()

# %%

df = pd.DataFrame({"t": t, "x": np.real(x), "y": np.abs(y)})
df.to_csv("pulse_compression.csv", index=False)
