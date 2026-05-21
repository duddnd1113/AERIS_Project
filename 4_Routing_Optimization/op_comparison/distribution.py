#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# =========================
# 1. Load CSV
# =========================
df = pd.read_csv("pm_filtered.csv")

print(df.head())
print(df.columns)

# If your csv has only one column, use this:
pm = df.iloc[:, 0].dropna().values

# If your column has a name, use this instead:
pm = df["재비산먼지 평균농도(㎍/㎥)"].dropna().values


# =========================
# 2. Remove invalid values
# =========================
pm = pm[pm >= 0]

# Box-Cox requires strictly positive values
pm_boxcox_input = pm + 1e-6


# =========================
# 3. Transform
# =========================
pm_log = np.log1p(pm)
pm_sqrt = np.sqrt(pm)
pm_boxcox, fitted_lambda = boxcox(pm_boxcox_input)

print(f"Box-Cox lambda: {fitted_lambda:.4f}")


# =========================
# =========================
# 4. Visualize
# =========================

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Original
axes[0, 0].hist(pm, bins=40, edgecolor="black")
axes[0, 0].set_title("Original PM Distribution")
axes[0, 0].set_xlabel("PM")
axes[0, 0].set_ylabel("Frequency")

# Log
axes[0, 1].hist(pm_log, bins=40, edgecolor="black")
axes[0, 1].set_title("Log Transform")
axes[0, 1].set_xlabel("log(1 + PM)")
axes[0, 1].set_ylabel("Frequency")

# Sqrt
axes[1, 0].hist(pm_sqrt, bins=40, edgecolor="black")
axes[1, 0].set_title("Sqrt Transform")
axes[1, 0].set_xlabel("sqrt(PM)")
axes[1, 0].set_ylabel("Frequency")

# Box-Cox
axes[1, 1].hist(pm_boxcox, bins=40, edgecolor="black")
axes[1, 1].set_title(f"Box-Cox Transform (lambda={fitted_lambda:.2f})")
axes[1, 1].set_xlabel("Box-Cox(PM)")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()

# VERY IMPORTANT
plt.show()
#%%
from scipy.stats import boxcox
import numpy as np
import matplotlib.pyplot as plt

lambdas = [-1, -0.5, 0, 0.25, 0.5, 1]

for l in lambdas:

    if l == 0:
        transformed = np.log(pm)
    else:
        transformed = (pm**l - 1) / l

    plt.hist(transformed, bins=40, alpha=0.5, label=f"λ={l}")

plt.legend()
plt.show()
# %%
