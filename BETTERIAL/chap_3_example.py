from pathlib import Path
from warnings import simplefilter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from scipy.signal import periodogram

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

""" Function """
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique())
    ax = sns.lineplot(x=freq, y=y, hue=period, data=X, ci=False, ax=ax, palette=palette, legend=False)
    ax.set_title(f"Seaonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(name, xy=(1, y_), xytext=(6, 0), color=line.get_color(), xycoords=ax.get_yaxis_transform(), textcoords="offset points", size=14, va="center")
    return ax

def plot_periodogram(ts, detrend='linear', ax=None):
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    frequncies, spectrum = periodogram(ts, fs=fs, detrend=detrend, window="boxcar", scaling="spectrum")
    if ax is None:
        _, ax = plt.subplots()
    ax.step(frequncies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)"
        ],
        rotation=30
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax

data_dir = Path("./input")
data = pd.read_csv(data_dir / "pressure_data.csv").drop("Time", axis=1).drop("Pressure2", axis=1).drop("Pressure3", axis=1).drop("pass", axis=1)

X = data.copy()
#
# # days within a week
# X["day"] = X.index.dayofweek
# X["week"] = X.index.week
#
# # days within a year
# X["dayofyear"] = X.index.dayofyear
# X["year"] = X.index.year
# fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 6))
# seasonal_plot(X, y="NumVehicles", period="week", freq="day", ax=ax0)
# seasonal_plot(X, y="NumVehicles", period="year", freq="dayofyear", ax=ax1)
# plot_periodogram(tunnel.NumVehicles, ax=ax2)
# plt.show()

fourier = CalendarFourier(freq="A", order=10)

dp = DeterministicProcess(index=data.index, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True)

X = dp.in_sample()
y = data["Pressure1"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(color="0.25", style=".", title = "P-HPH-01 Pressure Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()
plt.show()