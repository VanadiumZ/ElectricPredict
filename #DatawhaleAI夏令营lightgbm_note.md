## Datawhale AI summer camp
The code file for this note is lighgbmtrial.ipynb
The problem we need to sovle, about electric demand forecasting, is a time series forecasting problem.

### Feature Processing
In order to apply machine learning models to this problem, new features besides 'dt', 'type' are required to be created.
So in the features processing part we choose Historical Feature Translation and Window Statistical Processing.

``` Python
# concat and sort
data = pd.concat([test, train], axis=0, ignore_index=True)
data = data.sort_values(['id', 'dt'], ascending=False).reset_index(drop=True)

# 历史平移
for i in range(10, 30):
    data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)

# 窗口统计
data[f'win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3
```
### Model Training
We choose lightgbm to perform a trial.
Different from the form I learned before, I use 'train' rather than 'fit' in the training part.
And I acquire a more precise control and setting on the parameters.

``` Python
# set parameters of lgb
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'min_child_weight': 5, # higher => more conservative, avoid overfitting
        'num_leaves': 2 ** 5, # higher => more precise, may cause overfitting
        'lambda_l2': 10,
        'feature_fraction': 0.8, # rate of sampling featrues
        'bagging_fraction': 0.8, # rate of sampling data
        'bagging_freq': 4, # bagging frequency
        'learning_rate': 0.05,
        'seed': 2024,
        'nthread': 16, # the usage of the cores
        'verbose': -1, # display of info of training-(-1 means without display)
    }
```

It's a highly effient way to execute the prediction but apparently not that accurately. So the following work will focus on stacking and other ways to imporve the model.
