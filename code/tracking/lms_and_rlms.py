# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%

# t = np.linspace(0, 1, 100)
# X = np.array([np.cos(4 * np.pi * t) * t, np.sin(4 * np.pi * t) * t]).T
# # X = X + np.random.randn(*X.shape) * 0.01
# plt.plot(X[:, 0], X[:, 1], "b.")
# plt.axis("equal")
# plt.show()


def ls(X, order=3):
    S = []
    Z = []
    Xp = []
    H = None
    for i in range(order, X.shape[0]):
        S.append(X[i - order : i, :].flatten())
        Z.append(X[i, :])
        if H is not None:
            Xp.append(S[-1] @ H.T)

        Smat = np.array(S)
        Zmat = np.array(Z)

        H = Zmat.T @ np.linalg.pinv(Smat.T)
    return np.array(Xp)


def lms(X, order=3, mu=0.01):
    H = np.zeros((2, 2 * order))
    Xp = []
    for i in range(order, X.shape[0]):
        s = X[i - order : i, :].flatten().reshape(-1, 1)
        z = X[i, :].flatten().reshape(-1, 1)
        Xp.append(H @ s)
        e = z - Xp[-1]
        H += mu * e @ s.T
    return np.array(Xp).squeeze()


def rlms(X, order=3, lam=0.8, delta=100):
    p = 2 * order
    H = np.zeros((2, p))
    P = delta * np.eye(p)

    Xp = []

    for i in range(order, X.shape[0]):
        s = X[i - order : i, :].flatten().reshape(-1, 1)  # (p,1)
        z = X[i, :].reshape(-1, 1)  # (2,1)

        # 预测
        y = H @ s  # (2,1)
        Xp.append(y.ravel())

        # 误差
        e = z - y  # (2,1)

        # RLS 增益（标量分母，数值更稳）
        denom = lam + (s.T @ P @ s)  # 标量
        K = (P @ s) / denom  # (p,1)

        # 参数更新（矩阵形式：H <- H + e K^T）
        H = H + (e @ K.T)  # (2,1)(1,p)->(2,p)

        # 协方差更新
        P = (P - K @ s.T @ P) / lam

    return np.array(Xp)


def kf_cv(X, order=2, dt=1.0, q=1e-3, r=1e-6):
    """
    二维恒速(CV)卡尔曼滤波：返回每步的“预测量测” y_pred = H x_{t|t-1}
    X: (N,2)
    return: Xp (N-order, 2)
    """
    N = X.shape[0]
    assert N > order >= 1

    # 状态: [x, y, vx, vy]^T  —— 全程用列向量 (·,1)
    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    # 过程/量测噪声
    dt2, dt3, dt4 = dt * dt, dt**3, dt**4
    q11 = dt4 / 4
    q13 = dt3 / 2
    q33 = dt2
    Q = q * np.array(
        [[q11, 0, q13, 0], [0, q11, 0, q13], [q13, 0, q33, 0], [0, q13, 0, q33]],
        dtype=float,
    )
    R = r * np.eye(2)

    # 用前 order 点估计初始速度
    t_idx = np.arange(order, dtype=float).reshape(-1, 1)
    A = np.hstack([np.ones_like(t_idx), t_idx])
    bx, kx = np.linalg.lstsq(A, X[:order, 0], rcond=None)[0]
    by, ky = np.linalg.lstsq(A, X[:order, 1], rcond=None)[0]

    xk = np.array(
        [[X[order - 1, 0]], [X[order - 1, 1]], [kx / dt], [ky / dt]], dtype=float
    )  # (4,1)
    Pk = np.eye(4) * 1e-2

    Xp = []
    I4 = np.eye(4)

    for i in range(order, N):
        # 预测
        x_pred = F @ xk  # (4,1)
        P_pred = F @ Pk @ F.T + Q

        y_pred = (H @ x_pred).ravel()  # (2,)
        Xp.append(y_pred)

        # 修正
        z = X[i, :].reshape(2, 1)  # (2,1)
        innov = z - H @ x_pred  # (2,1)
        S = H @ P_pred @ H.T + R  # (2,2)
        K = P_pred @ H.T @ np.linalg.inv(S)  # (4,2)
        xk = x_pred + K @ innov  # (4,1)
        Pk = (I4 - K @ H) @ P_pred
    return np.array(Xp)


# %%
t1 = np.linspace(0, 1, 20)
t2 = np.linspace(1, 2, 20)
t3 = np.linspace(2, 3, 20)
x1 = t1
y1 = np.zeros_like(t1)
x2 = 2 - np.cos(np.pi * (t2 - 1))
y2 = np.sin(np.pi * (t2 - 1))

x3 = t3 + 1
y3 = (t3 - 2) ** 2

x = np.concatenate([x1, x2, x3])
y = np.concatenate([y1, y2, y3])
X = np.vstack([x, y]).T  # (N,2)

order = 4
Xls = rlms(X, order, lam=1, delta=1e10)
Xlms = lms(X, order, mu=4e-2)
Xrlms = rlms(X, order, lam=0.9, delta=100)
Xkf = kf_cv(X, order=order, dt=1.0, q=1e-6, r=1e-7)

# 可视化
plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], "b.", label="True")
plt.plot(Xls[:, 0], Xls[:, 1], "g.", label="LS pred")
plt.plot(Xlms[:, 0], Xlms[:, 1], "m.", label="LMS pred")
plt.plot(Xrlms[:, 0], Xrlms[:, 1], "r.", label="RLS pred")
plt.plot(Xkf[:, 0], Xkf[:, 1], "c.", label="KF pred")
plt.axis("equal")
plt.legend()
plt.show()
df = pd.DataFrame(
    {
        "x": X[:, 0],
        "y": X[:, 1],
    }
)
df.to_csv("tracking_1.csv", index=False)

df = pd.DataFrame(
    {
        "x1": Xls[:, 0],
        "y1": Xls[:, 1],
        "x2": Xlms[:, 0],
        "y2": Xlms[:, 1],
        "x3": Xrlms[:, 0],
        "y3": Xrlms[:, 1],
        "x4": Xkf[:, 0],
        "y4": Xkf[:, 1],
    }
)
df.to_csv("tracking_1_.csv", index=False)

# %%
t = np.linspace(0, 1, 100)
X = np.array([np.cos(4 * np.pi * t) * t, np.sin(4 * np.pi * t) * t]).T
# X = X + np.random.randn(*X.shape) * 0.01

order = 4
Xls = rlms(X, order, lam=1, delta=1e10)
Xlms = lms(X, order, mu=8e-1)
Xrlms = rlms(X, order, lam=0.999, delta=10)
Xkf = kf_cv(X, order=order, dt=1.0, q=1e-6, r=1e-8)

# 可视化
plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], "bo", label="True")
plt.plot(Xls[:, 0], Xls[:, 1], "g.", label="LS pred")
plt.plot(Xlms[:, 0], Xlms[:, 1], "m.", label="LMS pred")
plt.plot(Xrlms[:, 0], Xrlms[:, 1], "r.", label="RLS pred")
plt.plot(Xkf[:, 0], Xkf[:, 1], "c.", label="KF pred")
plt.axis("equal")
plt.legend()
plt.show()

df = pd.DataFrame(
    {
        "x": X[:, 0],
        "y": X[:, 1],
    }
)
df.to_csv("tracking_2.csv", index=False)

df = pd.DataFrame(
    {
        "x1": Xls[:, 0],
        "y1": Xls[:, 1],
        "x2": Xlms[:, 0],
        "y2": Xlms[:, 1],
        "x3": Xrlms[:, 0],
        "y3": Xrlms[:, 1],
        "x4": Xkf[:, 0],
        "y4": Xkf[:, 1],
    }
)
df.to_csv("tracking_2_.csv", index=False)

# %%

# %%
t1 = np.linspace(0, 1, 20)
t2 = np.linspace(1, 2, 20)
t3 = np.linspace(2, 3, 20)
x1 = t1
y1 = np.zeros_like(t1)
x2 = 2 - np.cos(np.pi * (t2 - 1))
y2 = np.sin(np.pi * (t2 - 1))

x3 = t3 + 1
y3 = (t3 - 2) ** 2

x = np.concatenate([x1, x2, x3])
y = np.concatenate([y1, y2, y3])
X = np.vstack([x, y]).T  # (N,2)
X = X + np.random.randn(*X.shape) * 0.02

order = 4
Xls = rlms(X, order, lam=1, delta=1e10)
Xlms = lms(X, order, mu=1e-2)
Xrlms = rlms(X, order, lam=0.9, delta=100)
Xkf = kf_cv(X, order=order, dt=1.0, q=1e-3, r=1e-2)

# 可视化
plt.figure(figsize=(6, 6))
plt.plot(X[:, 0], X[:, 1], "b.", label="True")
# plt.plot(Xls[:, 0], Xls[:, 1], "g.", label="LS pred")
# plt.plot(Xlms[:, 0], Xlms[:, 1], "m.", label="LMS pred")
# plt.plot(Xrlms[:, 0], Xrlms[:, 1], "r.", label="RLS pred")
plt.plot(Xkf[:, 0], Xkf[:, 1], "c.", label="KF pred")
plt.axis("equal")
plt.legend()
plt.show()
# %%
df = pd.DataFrame(
    {
        "x": X[:, 0],
        "y": X[:, 1],
    }
)
df.to_csv("tracking_3.csv", index=False)

df = pd.DataFrame(
    {
        "x1": Xls[:, 0],
        "y1": Xls[:, 1],
        "x2": Xlms[:, 0],
        "y2": Xlms[:, 1],
        "x3": Xrlms[:, 0],
        "y3": Xrlms[:, 1],
        "x4": Xkf[:, 0],
        "y4": Xkf[:, 1],
    }
)
df.to_csv("tracking_3_.csv", index=False)
