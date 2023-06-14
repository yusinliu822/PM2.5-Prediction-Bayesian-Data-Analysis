import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from tqdm import tqdm
import arviz as az
import pickle
from sklearn.metrics import mean_squared_error

siteid = 1

# read data
df_pm25 = pd.read_csv('data/pm25_processed.csv', low_memory=False)

# drop duplicates and set SiteID
df_pm25_part = df_pm25[df_pm25.SiteId == siteid]
df_pm25_part.drop_duplicates("PublishTime", inplace=True)

# make publish to index
df_pm25_part = df_pm25_part.set_index('PublishTime', inplace=False)
df_pm25_part.index = pd.to_datetime(df_pm25_part.index)

# choose the time
df_pm25_part = df_pm25_part[df_pm25_part.index.strftime('%Y-%m-%d') < '2023-1-1']

# setting future 1 hour, 2 hour, 3 hour pm2.5
df_pm25_part['PM1hour'] = df_pm25_part['PM2.5'].shift(periods=-1)
df_pm25_part['PM2hour'] = df_pm25_part['PM2.5'].shift(periods=-2)
df_pm25_part['PM3hour'] = df_pm25_part['PM2.5'].shift(periods=-3)

#use ffiil to fill the nan value
df_pm25_part = df_pm25_part.fillna(method='ffill')

# drop useless columns
df_pm25_part.drop(columns=['WindSpeed', 'WindDirec', 'Pollutant'], inplace=True)
df_pm25_part.to_json(f'data/pm25_changed{siteid}.json', indent=4)

# Plot coefficient relationship heatmap
fig, ax = plt.subplots(figsize=(20, 8))
sns.heatmap(df_pm25_part.corr(), annot=True, ax=ax, cmap='coolwarm')
plt.savefig(f'images/heatmap_region{siteid}.png')

X1 = df_pm25_part['PM10'].to_numpy()
X2 = df_pm25_part['AQI'].to_numpy()
X3 = df_pm25_part['PM10_AVG'].to_numpy()
X4 = df_pm25_part['PM2.5_AVG'].to_numpy()
X5 = df_pm25_part['PM2.5'].to_numpy()
Y1 = df_pm25_part['PM1hour'].to_numpy()
Y2 = df_pm25_part['PM2hour'].to_numpy()
Y3 = df_pm25_part['PM3hour'].to_numpy()

with pm.Model() as pm25_model:
    alpha1 = pm.Normal('alpha1', mu=0, sd=10)
    alpha2 = pm.Normal('alpha2', mu=0, sd=10)
    alpha3 = pm.Normal('alpha3', mu=0, sd=10)
    alpha4 = pm.Normal('alpha4', mu=0, sd=10)
    alpha5 = pm.Normal('alpha5', mu=0, sd=10)
    c = pm.Normal('c', mu=0, sd=30)
    
    sigma = pm.HalfCauchy('sigma', beta=10)
    
    mu1 = alpha1 * X1 + alpha2 * X2 + alpha3 * X3 + alpha4 * X4 + alpha5 * X5 + c
    mu2 = alpha1 * X1 + alpha2 * X2 + alpha3 * X3 + alpha4 * X4 + alpha5 * X5 + c
    mu3 = alpha1 * X1 + alpha2 * X2 + alpha3 * X3 + alpha4 * X4 + alpha5 * X5 + c
    
    Y_obs1 = pm.Normal('Y_obs1', mu=mu1, sd=sigma, observed=Y1)
    Y_obs2 = pm.Normal('Y_obs2', mu=mu2, sd=sigma, observed=Y2)
    Y_obs3 = pm.Normal('Y_obs3', mu=mu3, sd=sigma, observed=Y3)

with pm25_model:
    trace = pm.sample(3000, tune=1500, target_accept=0.9)

with pm25_model:
    posterior_pred = pm.sample_posterior_predictive(trace)
    prior = pm.sample_prior_predictive()
    y_pred_1hour = posterior_pred['Y_obs1'].mean(axis=0).round(1)
    y_pred_2hour = posterior_pred['Y_obs2'].mean(axis=0).round(1)
    y_pred_3hour = posterior_pred['Y_obs3'].mean(axis=0).round(1)

with pm25_model:
    data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_pred)


# pm25_model.save(f"models/model{siteid}")
az.plot_posterior(data, round_to=2, hdi_prob=0.95)
plt.savefig(f'images/posterior_plot{siteid}.png')

az.plot_trace(data, figsize=(10, 35))
plt.savefig(f'images/trace_plot{siteid}.png')

# 将预测值添加到DataFrame中
df_pm25_part['Y_pred_1hour'] = y_pred_1hour
df_pm25_part['Y_pred_2hour'] = y_pred_2hour
df_pm25_part['Y_pred_3hour'] = y_pred_3hour

df_pm25_part['prediction'] = df_pm25_part['Y_pred_1hour'].astype(str) + ' ' + df_pm25_part['Y_pred_2hour'].astype(str) + ' ' + df_pm25_part['Y_pred_3hour'].astype(str)

df_pm25_part[["SiteId", 'PM2.5', 'prediction']].to_csv('data/pm25_prediction2.csv', index=True)

for i in range(1,4):
    df_pm25_part[f'PM{i}hour'].plot()
    df_pm25_part[f'Y_pred_{i}hour'].plot()
    plt.legend()
    plt.savefig(f'images/prediction{i}hr{siteid}.png')
    plt.show()

    df_pm25_part[f'error{i}'] = df_pm25_part.apply(lambda x: abs(x[f'PM{i}hour']-x[f'Y_pred_{i}hour'])/x[f'PM{i}hour'] if x[f'PM{i}hour']!= 0 else np.mean(x[f'PM{i}hour']), axis=1)

df_pm25_part['error3'].plot()
df_pm25_part['error2'].plot()
df_pm25_part['error1'].plot()
plt.legend()
plt.savefig(f'images/error.png')

df_pm25_part.dropna(inplace=True)
# 從 DataFrame 中獲取 true label 和 prediction 列轉換為 NumPy 陣列
true_label1 = df_pm25_part['PM1hour'].to_numpy()
prediction1 = df_pm25_part['Y_pred_1hour'].to_numpy()
true_label2 = df_pm25_part['PM2hour'].to_numpy()
prediction2 = df_pm25_part['Y_pred_2hour'].to_numpy()
true_label3 = df_pm25_part['PM3hour'].to_numpy()
prediction3 = df_pm25_part['Y_pred_3hour'].to_numpy()

# 計算 MSE
mse_score1 = mean_squared_error(true_label1, prediction1)
mse_score2 = mean_squared_error(true_label2, prediction2)
mse_score3 = mean_squared_error(true_label3, prediction3)
print("MSE Score1:", mse_score1)
print("MSE Score2:", mse_score2)
print("MSE Score3:", mse_score3)
print("MSE Score:", (mse_score1 + mse_score2 + mse_score3)/3)