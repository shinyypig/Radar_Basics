# %%
import plotly.graph_objects as go
import numpy as np
import matplotlib.image as mpimg
import matplotlib


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


def receive(t):
    kappa = 100
    fc = 1e4
    return (
        rect(t)
        * np.exp(1j * 2 * np.pi * kappa * t**2)
        * np.exp(1j * 2 * np.pi * fc * t)
    )


t = np.linspace(-1, 1, 1000)
r = receive(t - 0.0)

dt = np.linspace(0, 1, 50) * (0)

X = []

for i in range(len(dt)):
    X.append(receive(t - dt[i]))

go.Figure(data=go.Heatmap(z=np.real(X))).show()

Z = np.real(X)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha

mpimg.imsave("array_angle_0.png", Z[:, ::-1, :], dpi=300, cmap="viridis")


# %%
dt = np.linspace(0, 1, 50) * (-1e-4)

X = []

for i in range(len(dt)):
    X.append(receive(t - dt[i]))

go.Figure(data=go.Heatmap(z=np.real(X))).show()

Z = np.real(X)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha

mpimg.imsave("array_angle_p.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%
dt = np.linspace(0, 1, 50) * (1e-4)

X = []

for i in range(len(dt)):
    X.append(receive(t - dt[i]))

go.Figure(data=go.Heatmap(z=np.real(X))).show()

Z = np.real(X)

Z = (Z - Z.min()) / (Z.max() - Z.min())

cmap = matplotlib.colormaps["viridis"]
Z = cmap(Z)[:, :, :3]  # 去掉 alpha

mpimg.imsave("array_angle_n.png", Z[:, ::-1, :], dpi=300, cmap="viridis")
