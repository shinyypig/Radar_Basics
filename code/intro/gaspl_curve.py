# %%
import numpy as np
import matplotlib.pyplot as plt
from pylayers.antprop.loss import gaspl
import pandas as pd

T = 15
PhPa = 1013
wvden = 7.5
d = 1000
fGHz = np.linspace(1, 1000, 1000)
L0 = gaspl(d, fGHz, T, PhPa, 0)
L0 = np.log10(L0) * 10  # Convert to dB for dry air
L = gaspl(d, fGHz, T, PhPa, wvden)
L = np.log10(L) * 10  # Convert to dB
plt.plot(fGHz, L)
plt.plot(fGHz, L0, "--", label="Dry Air")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Loss (dB)")
plt.title("Gaspl Loss vs Frequency")
plt.grid()
plt.show()

# %%
df = pd.DataFrame({"f": fGHz, "l": L, "l0": L0})
df.to_csv("img/intro/gaspl.csv", index=False)
