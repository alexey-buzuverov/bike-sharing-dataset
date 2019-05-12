# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

# Read .csv file
all = pd.read_csv('input/hour.csv', header=0)

# Feature Engineering
all.insert(5,'day',0)
all['day'] = all['dteday'].str[-2:].astype(int)
all['dteday'] = all['dteday'].astype('category').cat.codes
all['temp'] = (all['temp']*41).round(3)
all['atemp'] = (all['atemp']*50).round(3)
all['hum'] = (all['hum']*100).astype(int)
all['windspeed'] = (all['windspeed']*67).astype(int)

ft_mod = [col for col in all.columns.values if col not in ['instant','dteday','casual','registered','cnt']]
ft_weather = ['weathersit', 'temp', 'atemp', 'hum', 'windspeed']
ft_time = ['season', 'yr', 'mnth', 'day', 'hr', 'holiday', 'weekday', 'workingday']

### Quick check of 2 learning algorithms ###

# Ridge regression with CV
ridge_feats = all[ft_mod]
scaler = StandardScaler()
ridge_feats[ft_weather] = scaler.fit_transform(all[ft_weather])
ridge_feats = pd.get_dummies(ridge_feats, columns = ['season','mnth','day','weekday','hr'])
# ridge_feats.to_csv('ridge_feats.csv')
ridge_cv = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 100, 1000, 10000], cv=5)
ridge_cv.fit(ridge_feats, all['cnt'])
print('Ridge best alpha: %.9f.' %ridge_cv.alpha_)
ridge_pred = ridge_cv.predict(ridge_feats)
ridge_mae = mean_absolute_error(all['cnt'], ridge_pred)
print('Train mae ridge: %.9f.' %ridge_mae)

# LightGBM with default parameters
lgb_reg = lgb.LGBMRegressor(boosting_type = 'gbdt', metric = 'l1', verbose = 1)
lgb_reg.fit(all[ft_mod], all['cnt'])
lgb_pred= lgb_reg.predict(all[ft_mod])
lgb_mae = mean_absolute_error(all['cnt'], lgb_pred)
print('Train mae lightgbm: %.9f.' %lgb_mae)
# ax = lgb.plot_tree(lgb_reg, tree_index=0, figsize=(60, 20), show_info=['split_gain'])
# plt.tight_layout()
# plt.savefig('0th tree lgbm.png')

### LightGBM with early stopping training ###

random_state = 2019
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
oof = all[['instant', 'cnt']]
oof['predict'] = 0
val_maes = []
feature_importance = pd.DataFrame()

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l1',
    #'max_depth' : -1,
    #'num_leaves': 31,
    #'learning_rate': 0.1,
    #'feature_fraction': 1.0,
    #'bagging_fraction': 1.0,
    #'bagging_freq': 0,
    #'min_data_in_leaf': 20,
    #'min_sum_hessian_in_leaf': 1e-3,
    'bagging_seed' : random_state,
    #'boost_from_average': 'true',
    'verbosity' : 1,
    'seed': random_state
}

# Per fold training
for fold, (trn_idx, val_idx) in enumerate(kf.split(all)):
    X_train, y_train = all.loc[trn_idx, ft_mod], all.loc[trn_idx,'cnt']
    X_valid, y_valid = all.loc[val_idx, ft_mod], all.loc[val_idx,'cnt']

    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_reg = lgb.train(lgb_params,
                            trn_data,
                            100000,
                            valid_sets=[trn_data, val_data],
                            early_stopping_rounds=1000,
                            verbose_eval=1000,
                            evals_result=evals_result
                            )
    p_valid = lgb_reg.predict(X_valid)

    fold_importance = pd.DataFrame()
    fold_importance['feature'] = ft_mod
    fold_importance['importance'] = lgb_reg.feature_importance()
    fold_importance['fold'] = fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    oof['predict'][val_idx] = p_valid
    val_score = mean_absolute_error(y_valid, p_valid)
    val_maes.append(val_score)

# Print MAE
mean_mae = np.mean(val_maes)
std_mae = np.std(val_maes)
all_mae = mean_absolute_error(oof['cnt'], oof['predict'])
print('Final model: mean mae: %.9f, std: %.9f, all mae: %.9f.' % (mean_mae, std_mae, all_mae))

# Feature importance visualization
pp = PdfPages('feature_importance.pdf')
matplotlib.rcParams.update({'font.size': 11})
plt.figure(figsize=(8.27,11.69),dpi=150)
sns.barplot(x='importance', y='feature', data=feature_importance.sort_values(by='importance', ascending=False) )
plt.suptitle('LightGBM Features (averaged over folds)', fontsize=11)
pp.savefig()
pp.close()