# %%
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# %%
# 定义输入信号
x = np.linspace(-20, 20, 201)
y = np.sinc(x)

# 1. 原始信号
fig_original = go.Figure(
    data=go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="blue", width=2),
        name="Original Sinc Function",
    )
)
fig_original.update_layout(
    title="Original Sinc Function",
    xaxis_title="x",
    yaxis_title="sinc(x)",
    showlegend=True,
)
fig_original.show()

# %%
# 2. 线性相位滤波
Y = np.fft.fft(y)
N = np.arange(len(Y))

# 线性相位
phase_linear = np.pi * ((N - len(Y) // 2) / (len(Y) // 2)) * 20
phase_linear = np.fft.ifftshift(phase_linear)
phase_linear[0] = 0  # 保证直流分量相位为0

# 构造线性相位全通滤波器
H_linear = np.exp(1j * phase_linear)
# 强制频谱共轭对称
H_linear[len(H_linear) // 2 + 1 :] = np.conj(H_linear[1 : len(H_linear) // 2 + 1][::-1])

# 应用线性相位滤波
Y_linear = Y * H_linear
y_linear = np.real(np.fft.ifft(Y_linear))

# 线性相位滤波后的信号
fig_linear = go.Figure(
    data=go.Scatter(
        x=x,
        y=y_linear,
        mode="lines",
        line=dict(color="green", width=2),
        name="Linear Phase Filtered Signal",
    )
)
fig_linear.update_layout(
    title="Signal after Linear Phase All-pass Filter",
    xaxis_title="x",
    yaxis_title="Output signal",
    showlegend=True,
)
fig_linear.show()

# %%
# 3. 非线性相位滤波
# 非线性相位（三次方）
phase_nonlinear = np.sin((N - len(Y) // 2) / (len(Y) // 2) * np.pi * 3) * 8
phase_nonlinear = np.fft.ifftshift(phase_nonlinear)
phase_nonlinear[0] = 0  # 保证直流分量相位为0

# 构造非线性相位全通滤波器
H_nonlinear = np.exp(1j * phase_nonlinear)
# 强制频谱共轭对称
H_nonlinear[len(H_nonlinear) // 2 + 1 :] = np.conj(
    H_nonlinear[1 : len(H_nonlinear) // 2 + 1][::-1]
)

# 应用非线性相位滤波
Y_nonlinear = Y * H_nonlinear
y_nonlinear = np.real(np.fft.ifft(Y_nonlinear))

# 非线性相位滤波后的信号
fig_nonlinear = go.Figure(
    data=go.Scatter(
        x=x,
        y=y_nonlinear,
        mode="lines",
        line=dict(color="red", width=2),
        name="Nonlinear Phase Filtered Signal",
    )
)
fig_nonlinear.update_layout(
    title="Signal after Nonlinear Phase All-pass Filter",
    xaxis_title="x",
    yaxis_title="Output signal",
    showlegend=True,
)
fig_nonlinear.show()

# %%
# 4. 频谱对比
freq = np.fft.fftfreq(len(Y))

fig_spectrum = go.Figure(
    data=[
        go.Scatter(
            x=freq,
            y=np.abs(Y),
            mode="lines",
            line=dict(color="blue", width=2),
            name="Original Spectrum",
        ),
        go.Scatter(
            x=freq,
            y=np.abs(Y_linear),
            mode="lines",
            line=dict(color="green", width=2),
            name="Linear Phase Spectrum",
        ),
        go.Scatter(
            x=freq,
            y=np.abs(Y_nonlinear),
            mode="lines",
            line=dict(color="red", width=2),
            name="Nonlinear Phase Spectrum",
        ),
    ]
)
fig_spectrum.update_layout(
    title="Frequency Spectrum Comparison",
    xaxis_title="Frequency",
    yaxis_title="Magnitude",
    showlegend=True,
)
fig_spectrum.show()

# %%
# 5. 时域信号对比
fig_time_comparison = go.Figure(
    data=[
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Original Signal",
        ),
        go.Scatter(
            x=x,
            y=y_linear,
            mode="lines",
            line=dict(color="green", width=2),
            name="Linear Phase Filtered",
        ),
        go.Scatter(
            x=x,
            y=y_nonlinear,
            mode="lines",
            line=dict(color="red", width=2),
            name="Nonlinear Phase Filtered",
        ),
    ]
)
fig_time_comparison.update_layout(
    title="Time Domain Signal Comparison",
    xaxis_title="x",
    yaxis_title="Amplitude",
    showlegend=True,
)
fig_time_comparison.show()

# %%
df = pd.DataFrame({"x": x, "y1": y, "y2": y_linear, "y3": y_nonlinear})
df.to_csv("img/intro/phase_signal.csv", index=False)

df = pd.DataFrame(
    {
        "x": x,
        "y1": np.fft.fftshift(np.abs(Y)),
        "y2": np.fft.fftshift(np.abs(Y_linear)),
        "y3": np.fft.fftshift(np.abs(Y_nonlinear)),
    }
)
df.to_csv("img/intro/phase_spectrum.csv", index=False)
