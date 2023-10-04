from src.preprocessing import deal_with_site
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from tqdm import tqdm
import arviz as az
from sklearn.metrics import mean_squared_error
from src.config import *
import datetime
import wandb

def model(df_pm25_part):

    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY)

    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")

    X1 = df_pm25_part['PM10'].to_numpy()
    X2 = df_pm25_part['AQI'].to_numpy()
    X3 = df_pm25_part['PM10_AVG'].to_numpy()
    X4 = df_pm25_part['PM2.5_AVG'].to_numpy()
    X5 = df_pm25_part['PM2.5'].to_numpy()
    Y1 = df_pm25_part['PM1hour'].to_numpy()
    Y2 = df_pm25_part['PM2hour'].to_numpy()
    Y3 = df_pm25_part['PM3hour'].to_numpy()

    X = np.column_stack((X1, X2, X3, X4, X5))

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

    gv = pm.model_to_graphviz(pm25_model)
    gv.render(filename=f'images/model{SITEID}_{current_time}',format='png')
    wandb.log({"model": wandb.Image(f'images/model{SITEID}_{current_time}.png')})

    az.plot_posterior(data, round_to=2, hdi_prob=0.95)

    df_pm25_part['Y_pred_1hour'] = y_pred_1hour
    df_pm25_part['Y_pred_2hour'] = y_pred_2hour
    df_pm25_part['Y_pred_3hour'] = y_pred_3hour

    df_pm25_part['prediction'] = df_pm25_part['Y_pred_1hour'].astype(str) + ' ' + df_pm25_part['Y_pred_2hour'].astype(str) + ' ' + df_pm25_part['Y_pred_3hour'].astype(str)

    df_pm25_part[["SiteId", 'PM2.5', 'prediction']].to_csv(f'data/pm25_prediction{SITEID}_{current_time}.csv', index=True)

    for i in range(1,4):

        df_pm25_part[f'error{i}'] = df_pm25_part.apply(lambda x: abs(x[f'PM{i}hour']-x[f'Y_pred_{i}hour'])/x[f'PM{i}hour'] if x[f'PM{i}hour']!= 0 else np.mean(x[f'PM{i}hour']), axis=1)

    df_pm25_part.dropna(inplace=True)
    
    # 從 DataFrame 中獲取 true label 和 prediction 列轉換為 NumPy 陣列
    true_label1 = df_pm25_part['PM1hour'].to_numpy()
    prediction1 = df_pm25_part['Y_pred_1hour'].to_numpy()
    true_label2 = df_pm25_part['PM2hour'].to_numpy()
    prediction2 = df_pm25_part['Y_pred_2hour'].to_numpy()
    true_label3 = df_pm25_part['PM3hour'].to_numpy()
    prediction3 = df_pm25_part['Y_pred_3hour'].to_numpy()

    # 計算 MSE
    mse_score1 = mean_squared_error(true_label1, prediction1).round(2)
    mse_score2 = mean_squared_error(true_label2, prediction2).round(2)
    mse_score3 = mean_squared_error(true_label3, prediction3).round(2)
    wandb.log({"mse_score1":mse_score1, "mse_score2":mse_score2, "mse_score3":mse_score3})
    print("MSE Score1:", mse_score1)
    print("MSE Score2:", mse_score2)
    print("MSE Score3:", mse_score3)
    print("MSE Score:", (mse_score1 + mse_score2 + mse_score3)/3)

if __name__ == '__main__':

    df_pm25_part = deal_with_site()
    model(df_pm25_part)