# PM2.5-Prediction-Bayesian-Data-Analysis

## Introduction

Based on the historical PM2.5 data and other potential features provided by the Environmental Protection Agency for 77 monitoring stations, we will build a model to predict PM2.5 concentration index for the next 3 hours.

## Data

1. Go to https://history.colife.org.tw/#/
2. Download data of in `空氣品質/環保署_國家空品測站`
3. We use data collected in January 2021 to October 2022
4. Put the zip files in `./data` and run

   ```
   sh unzip.sh
   ```

## Environment

Create conda environment

```
conda create -n env_name python==3.8
conda activate env_name
```

Install packages

```
pip install numpy==1.22 arviz jupyter theano-pymc scipy==1.7.3 arviz==0.12.1 seaborn pymc3 scikit-learn tqdm wandb

conda install -c conda-forge python-graphviz
```

Run the code

```
python main.py
```

## Preprocessing

The downloaded data consists of 22 files, each representing data for a specific month, with file names in the format `EPA_OD_YYYYMM.csv`. Integrate all data files into a unified dataset, examine and address its characteristics and missing values.

`label_dict.json`: [ ColumnOrder, Datatype, PollutantMapping ]
`pm2.5.csv`: (Generate)Integrate all data files into a unified dataset.
`pm25_processed.csv`: Finish preprocessing.

## Model

Correlation heatmap:
![Correlation heatmap](https://github.com/yusinliu822/PM2.5-Prediction-Bayesian-Data-Analysis/blob/Winnie/images/heatmap_region32.png?raw=true)

Based on the observation of the correlation heatmap, we have identified certain columns that exhibit a higher correlation with the PM2.5 values at these time points. It is presumed that these columns are likely to be the primary influencing factors on PM2.5. We have selected columns such as `'PM10,' 'AQI,' 'PM10_AVG,' 'PM2.5_AVG,', 'PM2.5'` as features for building the model.

Model Formula:

```math
PM_{2.5}{ }^k = \text{Background} + \sum_{i=0}^{5} \alpha_i x_i \cdot \text{for k} = 1hr, 2hr, 3hr
```

Model structure:

![Model structure](https://github.com/yusinliu822/PM2.5-Prediction-Bayesian-Data-Analysis/blob/Winnie/images/model.png?raw=true)

## Result

Example setting :  `SITEID == 32`

```code
MSE Score1: 11.808378677349076
MSE Score2: 18.262630873568064
MSE Score3: 23.502010021786493
MSE Score: 17.85767319090121
```

Posterior
![Posterior](https://github.com/yusinliu822/PM2.5-Prediction-Bayesian-Data-Analysis/blob/Winnie/images/posterior_plot32.png?raw=true)
Trace plot
![Trace plot](https://github.com/yusinliu822/PM2.5-Prediction-Bayesian-Data-Analysis/blob/Winnie/images/trace_plot32.png?raw=true)
Error
![Error](https://github.com/yusinliu822/PM2.5-Prediction-Bayesian-Data-Analysis/blob/Winnie/images/error32.png?raw=true)
