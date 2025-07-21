# %%
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

# 读取图像
img = Image.open("img/matrix/cat.jpg")
# resize to 600 * 400
img = img.resize((300, 200))

# save the resized image, gray scale
img = img.convert("L")
img.save("img/matrix/cat_resized.jpg")

img_array = np.array(img)

# 如果是彩色图像，转换为灰度图像
if len(img_array.shape) == 3:
    img_gray = np.mean(img_array, axis=2)
else:
    img_gray = img_array

# 进行SVD分解
U, s, Vt = np.linalg.svd(img_gray, full_matrices=True)

print(f"原始图像形状: {img_gray.shape}")
print(f"U形状: {U.shape}")
print(f"奇异值个数: {len(s)}")
print(f"Vt形状: {Vt.shape}")

go.Figure(
    go.Scatter(
        x=np.arange(len(s)),
        y=s,
        mode="markers+lines",
        name="奇异值",
        marker=dict(size=5, color="blue"),
    )
).update_layout(
    title="奇异值分解的奇异值",
    xaxis_title="奇异值索引",
    yaxis_title="奇异值",
    # yaxis=dict(type="log"),
    template="plotly_white",
).show()

df = pd.DataFrame({"x": 1 + np.arange(len(s)), "y": s})
df.to_csv("img/matrix/singular_values.csv", index=False)

# %%
# 仅保留前10个奇异值的图像
k = 10
S_k = np.zeros_like(img_gray, dtype=float)
S_k[:k, :k] = np.diag(s[:k])
img_approx = U @ S_k @ Vt
fig = go.Figure(data=go.Heatmap(z=img_approx, colorscale="gray")).show()

img_approx = np.clip(img_approx, 0, 255).astype(np.uint8)
Image.fromarray(img_approx).save("img/matrix/cat_approx1.jpg")

k = 20
S_k = np.zeros_like(img_gray, dtype=float)
S_k[:k, :k] = np.diag(s[:k])
img_approx = U @ S_k @ Vt
fig = go.Figure(data=go.Heatmap(z=img_approx, colorscale="gray")).show()

img_approx = np.clip(img_approx, 0, 255).astype(np.uint8)
Image.fromarray(img_approx).save("img/matrix/cat_approx2.jpg")


k = 50
S_k = np.zeros_like(img_gray, dtype=float)
S_k[:k, :k] = np.diag(s[:k])
img_approx = U @ S_k @ Vt
fig = go.Figure(data=go.Heatmap(z=img_approx, colorscale="gray")).show()
img_approx = np.clip(img_approx, 0, 255).astype(np.uint8)
Image.fromarray(img_approx).save("img/matrix/cat_approx3.jpg")
