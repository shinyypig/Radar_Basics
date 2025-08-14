# %%
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

# 读取图像
img = Image.open("img/matrix/cat.jpg")
# resize to 600 * 400
img = img.resize((300, 200))


img = np.array(img)

go.Figure(go.Image(z=img)).update_layout(
    title="原始图像",
    xaxis_title="宽度",
    yaxis_title="高度",
    template="plotly_white",
).show()


img_r = img[:, :, 0]
img_g = img[:, :, 1]
img_b = img[:, :, 2]

# save the three channels as images
Image.fromarray(img_r).save("img/matrix/cat_r.jpg")
Image.fromarray(img_g).save("img/matrix/cat_g.jpg")
Image.fromarray(img_b).save("img/matrix/cat_b.jpg")

# %%
# mode-k unfolding

m2 = img.transpose(0, 2, 1).reshape(200, 300 * 3)
m1 = img.transpose(1, 2, 0).reshape(300, 200 * 3)

# save the two matrices as image

Image.fromarray(m1.astype(np.uint8)).save("img/matrix/cat_m1.jpg")
Image.fromarray(m2.astype(np.uint8)).save("img/matrix/cat_m2.jpg")
