import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from warnings import simplefilter
from sklearn.linear_model import LinearRegression

""" Book_sales """
# df = pd.read_csv("./input/ts-course-data/book_sales.csv", index_col='Date', parse_dates=['Date']).drop('Hardcover', axis=1)
# df['Time'] = np.arange(len(df.index))
#
# # plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True, figsize=(11,4), titlesize=18, titleweight='bold')
# plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=16, titlepad=10)
# # print(df.head())
# #
# #
# # #### Check the plot based on time ###
# # fig, ax = plt.subplots()
# # ax.plot('Time', 'Hardcover', data=df, color='0.75')
# # ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
# # ax.set_title('Time Plot of Hardcover Sales')
# # plt.show()
#
#
# #### Check the plot based on Lag ###
# df['Lag_1'] = df['Paperback'].shift(5)
# df = df.reindex(columns=['Paperback', 'Lag_1'])
# print(df.head())
#
# fig, ax = plt.subplots()
# ax = sns.regplot(x='Lag_1', y='Paperback', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_aspect('equal')
# ax.set_title('Lag Plot of Hardcover Sales')
# plt.show()

""" Tunnel Traffic """
simplefilter("ignore")

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)
plot_params = dict(color="0.75", style=".-", markeredgecolor="0.25", markerfacecolor="0.25", legend=False)

tunnel = pd.read_csv("./input/ts-course-data/tunnel.csv", index_col='Day', parse_dates = ['Day'])

df = tunnel.copy()
df['Time'] = np.arange(len(df.index))

# X = df.loc[:, ['Time']]
# y = df.loc[:, 'NumVehicles']
#
# # Train the model
# model = LinearRegression()
# model.fit(X, y)
#
# y_pred = pd.Series(model.predict(X), index=X.index)
#
# ax = y.plot(**plot_params)
# ax = y_pred.plot(ax=ax, linewidth=3)
# ax.set_title('Time plot of Tunnel Traffic')


### Lag plot ###
df['Lag_1'] = df['NumVehicles'].shift(1)

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)
y = df.loc[:, 'NumVehicles']
y, X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag plot of Tunnel Traffic')

plt.show()