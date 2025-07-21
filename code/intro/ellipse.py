# %%
import numpy as np
import plotly.graph_objects as go

theta = np.linspace(0, 2 * np.pi, 200)

A = np.matrix([[2, 1], [1, 2]])

V = np.matrix([np.cos(theta), np.sin(theta)])
k = np.sum(np.multiply(A @ V, V), axis=0)

V = np.divide(V, np.sqrt(k))

fig = go.Figure(data=go.Scatter(x=V[0, :].A1, y=V[1, :].A1, mode="lines"))

# calculate the eigenvectors of V
eigvals, eigvecs = np.linalg.eig(A)

fig.add_trace(
    go.Scatter(
        x=eigvecs[0, :].A1,
        y=eigvecs[1, :].A1,
        mode="markers",
        marker=dict(size=10, color="red"),  # red color for eigenvectors
        name="Eigenvectors",
    )
)

fig.update_layout(
    xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1)
)

fig.show()
