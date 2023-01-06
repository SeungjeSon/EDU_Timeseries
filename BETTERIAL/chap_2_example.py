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

data_dir = Path("./input")
data = pd.read_csv(data_dir / "220906_PT_A_99_0.csv").drop("Time", axis=1).drop("Pressure2", axis=1).drop("Pressure3", axis=1).drop("pass", axis=1)
# data['Time'] = np.arange(len(data.index))

drop_params = data[data.Pressure1 <10].index
data.drop(drop_params, inplace=True)
# print(data.head())

dp = DeterministicProcess(
    index=data.index,
    constant=True,
    order=1
)
X = dp.in_sample()
y = data["Pressure1"]

ax = data.plot(style=".", color="0.5")
plt.show()