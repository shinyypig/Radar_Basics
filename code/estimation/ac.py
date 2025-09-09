# %%
import numpy as np
import plotly.graph_objects as go
import pandas as pd

M = 5
theta = np.linspace(-np.pi / 30, np.pi / 30, 100)
r = -np.tan(np.pi * M * np.sin(theta) / 2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=theta * 180 / np.pi, y=r, mode="lines", name="虚部比值"))
fig.update_layout(
    title="和差比幅测角中的虚部比值",
    xaxis_title="方位角 θ (度)",
    yaxis_title="虚部比值 r",
    # yaxis=dict(range=[-10, 10]),
    template="plotly_white",
)
fig.show()

df = pd.DataFrame({"x": theta * 180 / np.pi, "y": r})
df.to_csv("ac.csv", index=False)
