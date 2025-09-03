# %%
import plotly.graph_objects as go
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.image as mpimg
import pandas as pd


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def s1(t, f):
    return rect(t) * np.exp(1j * np.pi * f * t)


def s2(t, kappa):
    return rect(t) * np.exp(1j * np.pi * kappa * t**2)


def s3(t, kappa):
    return rect(t) * np.exp(1j * np.pi * kappa * t**3)


t = np.linspace(-1, 1, 1000)

kappa = 100
x1 = s1(t, 20)
x2 = s2(t, 20)
x3 = s2(t, 100)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.real(x1), mode="lines", name="Signal 1"))
fig.add_trace(go.Scatter(x=t, y=np.real(x2), mode="lines", name="Signal 2"))
fig.add_trace(go.Scatter(x=t, y=np.real(x3), mode="lines", name="Signal 3"))
fig.show()

# x1 = np.abs(x1)

r1 = signal.correlate(x1, x1, mode="same", method="direct")
r2 = signal.correlate(x2, x2, mode="same", method="direct")
r3 = signal.correlate(x3, x3, mode="same", method="direct")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=np.abs(r1), mode="lines", name="Result 1"))
fig.add_trace(go.Scatter(x=t, y=np.abs(r2), mode="lines", name="Result 2"))
fig.add_trace(go.Scatter(x=t, y=np.abs(r3), mode="lines", name="Result 3"))
fig.show()

df = pd.DataFrame(
    {
        "t": t,
        "x1": np.real(x1),
        "x2": np.real(x2),
        "x3": np.real(x3),
        "r1": np.abs(r1),
        "r2": np.abs(r2),
        "r3": np.abs(r3),
    }
)
df.to_csv("pulse_compression_var.csv", index=False)
