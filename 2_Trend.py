from pathlib import Path
from warnings import simplefilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# Load Tunnel Traffic dataset
data_dir = Path("./input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period()

# # 1년기준 이동평균
# moving_avg = tunnel.rolling(window=365, center=True, min_periods=183).mean()
#
# ax = tunnel.plot(style=".", color='0.5')
# moving_avg.plot(ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False)
#
# plt.show()

""" using DeterministicProcess """
dp = DeterministicProcess(
    index=tunnel.index,
    constant=True,
    order=1,
    drop=True
)
X = dp.in_sample()

y = tunnel["NumVehicles"]

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

# ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
# _ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
# plt.show()

X = dp.out_of_sample(steps=30)
y_fore = pd.Series(model.predict(X), index=X.index)
ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, color="C3", label="Trend Forecast")
_ = ax.legend()
plt.show()