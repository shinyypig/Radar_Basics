# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd


def plot_roc(
    mu0=0.0, sigma0=1.0, mu1=1.0, sigma1=1.0, num_points=400, savepath=None, show=True
):
    """
    绘制两正态分布 H0 ~ N(mu0, sigma0^2) 和 H1 ~ N(mu1, sigma1^2) 的 ROC 曲线，
    返回计算得到的 AUC。
    """
    if sigma0 <= 0 or sigma1 <= 0:
        raise ValueError("sigma0 和 sigma1 必须大于 0")

    # 根据分布范围自动选择阈值区间
    lo = min(mu0 - 6 * sigma0, mu1 - 6 * sigma1)
    hi = max(mu0 + 6 * sigma0, mu1 + 6 * sigma1)
    gammas = np.linspace(lo, hi, num_points)

    # P_F = P(x > gamma | H0), P_D = P(x > gamma | H1)
    P_F = norm.sf(gammas, loc=mu0, scale=sigma0)
    P_D = norm.sf(gammas, loc=mu1, scale=sigma1)

    # 为计算 AUC 按 P_F 升序排列
    order = np.argsort(P_F)
    P_F_sorted = P_F[order]
    P_D_sorted = P_D[order]
    auc = np.trapz(P_D_sorted, P_F_sorted)

    plt.figure(figsize=(6, 6))
    plt.plot(P_F, P_D, lw=2, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("P_F")
    plt.ylabel("P_D")
    plt.title(f"ROC: H0 ~ N({mu0},{sigma0**2}), H1 ~ N({mu1},{sigma1**2})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return P_F_sorted, P_D_sorted, auc


# 使用示例
if __name__ == "__main__":
    P_F1, P_D1, auc1 = plot_roc(mu0=0, sigma0=1, mu1=1, sigma1=1, num_points=200)
    P_F2, P_D2, auc2 = plot_roc(mu0=0, sigma0=1, mu1=2, sigma1=1, num_points=200)
    P_F3, P_D3, auc3 = plot_roc(mu0=0, sigma0=1, mu1=3, sigma1=1, num_points=200)
    P_F4, P_D4, auc4 = plot_roc(mu0=0, sigma0=1, mu1=4, sigma1=1, num_points=200)
    df = pd.DataFrame(
        {
            "x1": P_F1,
            "y1": P_D1,
            "x2": P_F2,
            "y2": P_D2,
            "x3": P_F3,
            "y3": P_D3,
            "x4": P_F4,
            "y4": P_D4,
        }
    )
    df.to_csv("roc_curves.csv", index=False)
