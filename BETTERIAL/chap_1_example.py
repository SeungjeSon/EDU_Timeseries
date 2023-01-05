import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:\SON\project\EDU_Timeseries\EDU_Timeseries\BETTERIAL\input/pressure_data.csv").drop('Time', axis=1).drop('pass', axis=1)
df['Time'] = np.arange(len(df.index))

# plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11,4), titlesize=18, titleweight='bold')
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=16, titlepad=10)
# print(df.head())


# #### Check the plot based on time ###
# fig, ax = plt.subplots()
# ax.plot('Time', 'Hardcover', data=df, color='0.75')
# ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_title('Time Plot of Hardcover Sales')
# plt.show()


#### Check the plot based on Lag ###
# df['Lag_1'] = df['Pressure1'].shift(1)
# # df['Lag_2'] = df['Pressure2'].shift(1)
# # df = df.reindex(columns=['Pressure1', 'Lag_1'])
# # df['Lag_3'] = df['Pressure3'].shift(1)
# df = df.reindex(columns=['Pressure1', 'Lag_1'])
# # df = df.reindex(columns=['Pressure2', 'Lag_2'])
# # df = df.reindex(columns=['Pressure3', 'Lag_3'])
# # print(df.head())
#
# fig, ax = plt.subplots()
# ax = sns.regplot(x='Lag_1', y='Pressure1', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_aspect('equal')
# ax.set_title('Lag Plot of Pressure1')
# plt.show()

### LinearRegression ###
df['Lag_1'] = df['Pressure1'].shift(1)
X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)
y = df.loc[:, 'Pressure1']
y, X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')

plt.show()