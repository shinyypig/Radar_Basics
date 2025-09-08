# %%
import numpy as np
import plotly.graph_objects as go
import pandas as pd


Mlist = [11, 21, 51]
A = []
w = np.linspace(-0.5, 0.5, 1000)
fig = go.Figure()
for M in Mlist:
    a = np.exp(1j * 2 * np.pi * w[:, None] * np.arange(M)[None, :])

    a = np.mean(a, axis=1)
    a = np.abs(a)
    A.append(a)

    fig.add_trace(
        go.Scatter(
            x=w,
            y=a,
            mode="lines",
            name=f"M={M}",
        )
    )
fig.show()

df = pd.DataFrame({"w": w, "a1": A[0], "a2": A[1], "a3": A[2]})
df.to_csv("weight.csv", index=False)
