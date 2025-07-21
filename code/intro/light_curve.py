# %%
import numpy as np
import matplotlib.pyplot as plt

theta0 = np.deg2rad(30)
n0 = 1.5
k = 1e-2
a = n0 * np.sin(theta0)

y = np.linspace(0, 200e3, 200)

expr = n0 + k * y
x = (1 / (k * a)) * (
    (expr / 2) * np.sqrt(expr**2 - a**2)
    - (a**2 / 2) * np.log(expr + np.sqrt(expr**2 - a**2))
)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Light ray trajectory in gradient-index medium")
plt.grid(True)
plt.show()
