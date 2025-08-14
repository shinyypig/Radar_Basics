# %%
import plotly.graph_objects as go
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.image as mpimg
import pandas as pd


def rect(t):
    return np.where(np.abs(t) <= 0.5, 1.0, 0.0)


t = np.linspace(-1, 1, 1000)

kappa = 100
x1 = rect(t) * np.exp(1j * np.pi * kappa * t**2)

df = pd.DataFrame({"t": t, "x": np.real(x1)})
df.to_csv("chirp.csv", index=False)

go.Figure(data=go.Scatter(x=t, y=np.real(x1), mode="lines", name="Real Part")).show()

# 假设已有 x1
fs = 500
f_stft, t_stft, Zxx = signal.stft(x1, fs=fs, nperseg=100, noverlap=90, nfft=256)

# 将频率搬到居中（零频在中间）
freqs = np.fft.fftshift(f_stft)
Zxx_shift = np.fft.fftshift(Zxx, axes=0)
Zxx_shift = abs(Zxx_shift)

vmin, vmax = Zxx_shift.min(), Zxx_shift.max()
Zxx_shift = (Zxx_shift - vmin) / (vmax - vmin)
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Zxx_shift)[:, :, :3]  # 去掉 alpha

tmin, tmax = float(t_stft.min()), float(t_stft.max())
fmin, fmax = float(freqs.min()), float(freqs.max())
print(f"tmin={tmin}, tmax={tmax}, fmin={fmin}, fmax={fmax}")

# 保存图像
mpimg.imsave("chirp_stft.png", Z[:, ::-1, :], dpi=300, cmap="viridis")

# %%

kappa = 100
x1 = rect(t) * np.exp(1j * np.pi * kappa * t**3)

df = pd.DataFrame({"t": t, "x": np.real(x1)})
df.to_csv("cubic.csv", index=False)

go.Figure(data=go.Scatter(x=t, y=np.real(x1), mode="lines", name="Real Part")).show()

# 假设已有 x1
fs = 500
f_stft, t_stft, Zxx = signal.stft(x1, fs=fs, nperseg=100, noverlap=90, nfft=256)

# 将频率搬到居中（零频在中间）
freqs = np.fft.fftshift(f_stft)
Zxx_shift = np.fft.fftshift(Zxx, axes=0)
Zxx_shift = abs(Zxx_shift)

vmin, vmax = Zxx_shift.min(), Zxx_shift.max()
Zxx_shift = (Zxx_shift - vmin) / (vmax - vmin)
cmap = matplotlib.colormaps["viridis"]
Z = cmap(Zxx_shift)[:, :, :3]  # 去掉 alpha

tmin, tmax = float(t_stft.min()), float(t_stft.max())
fmin, fmax = float(freqs.min()), float(freqs.max())
print(f"tmin={tmin}, tmax={tmax}, fmin={fmin}, fmax={fmax}")

# 保存图像
mpimg.imsave("cubic_stft.png", Z[:, ::-1, :], dpi=300, cmap="viridis")


# %%

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------- 1) 生成 Barker-13 脉冲 ----------
# 芯片序列（+1/-1）
barker13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1], dtype=float)
L = len(barker13)


# 每个码芯片 50 个点
samples_per_chip = 50

# 重复每个码值
waveform = np.repeat(barker13, samples_per_chip)

# 前后各加 100 个 0
waveform = np.concatenate([np.zeros(500), waveform, np.zeros(500)])

# 时间轴
t = np.linspace(-1, 1, len(waveform))

df = pd.DataFrame({"t": t, "x": waveform})
df.to_csv("barker13.csv", index=False)

# 绘制
plt.figure(figsize=(10, 3))
plt.plot(t, waveform, drawstyle="steps-post", linewidth=2)
plt.title("Barker-13 波形（每码 50 个点）")
plt.xlabel("采样点")
plt.ylabel("幅度")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()
