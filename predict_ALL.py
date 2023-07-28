import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import comb

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, BayesianRidge
from sklearn.linear_model import QuantileRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel as C,
                                              Matern, WhiteKernel, DotProduct)
import xgboost
from ngboost import NGBRegressor
from catboost import CatBoostRegressor
from ngboost.distns import Exponential, Normal

from sklearn.utils import shuffle


from statsmodels.stats.outliers_influence import variance_inflation_factor



def pre_progressing(original_data):
    data = original_data[:, :-1]
    target = original_data[:, -1]
    normalize_data = MinMaxScaler().fit_transform(data)
    log_target = np.log(target.astype('float'))
    normalize_target = MinMaxScaler().fit_transform(log_target)
    
    return normalize_data, log_target, normalize_target

def load_data(data_dir,fake_dir):
    df = pd.read_csv(data_dir)
    #sample_data = df.values[:, 2:]
    #columns = df.columns[2:]
    #normalize_feature, log_target = pre_progressing(sample_data)
    #data = pd.DataFrame(np.concatenate([normalize_feature, log_target.reshape(692, 1)], axis=1), columns=columns)
    #data = data.drop(['4TTi '], axis=1)
    #data = data.drop(['4TTe'], axis=1)
    X = df[list(df.columns[:-1])]
    y = df['RT']
    data = df

    df_f = pd.read_csv(fake_dir)
    X_f = df_f[list(df_f.columns[:-1])]
    y_f = df_f['RT']
    
    #real_data and fake data
    data_fake = pd.concat([data, df_f])
    data_fake = shuffle(data_fake)
    return X,y,data,X_f,y_f,data_fake

def train_test(data_all,eps):
    data_train, data_test = train_test_split(data_all, train_size=eps,
                                       shuffle=True,random_state=None)
    X_train,X_test = data_train[list(data_train.columns[:-1])],data_test[list(data_test.columns[:-1])]
    y_train,y_test = data_train['RT'],data_test['RT']
    return X_train,X_test,y_train,y_test



def train_simple_model(method,X_train,X_test,y_train):
    if method=='Linear Regression':
        model = LinearRegression()
        model.fit(X_train.values, y_train.values)
    if method=='Lasso Regression':
        model = Lasso(0.00390625)
        model.fit(X_train.values, y_train.values)
    if method=='Bayesian Ridge Regression':
        model = BayesianRidge(alpha_1=72.12096988,alpha_2=228.55120009,lambda_1=33.32712824,lambda_2=256)
        #model = BayesianRidge(alpha_1=96.04101504, alpha_2=256, lambda_1=35, lambda_2=256)
        #model = BayesianRidge(alpha_1=3.90625000e-03, alpha_2=2.26653768e+02, lambda_1=2.55582038e+01, lambda_2=256)
        model.fit(X_train.values, y_train.values)
    if method=='Quantile Regression':
        quantiles = [0.025, 0.5, 0.975]
        for q in quantiles:
            if q == 0.025:
                model_L = QuantileRegressor(quantile=q, alpha=0.001)
                model_L.fit(X_train.values, y_train.values)
            if q == 0.5:
                model = QuantileRegressor(quantile=q, alpha=0.001)
                model.fit(X_train.values, y_train.values)
            if q == 0.975:
                model_U = QuantileRegressor(quantile=q, alpha=0.001)
                model_U.fit(X_train.values, y_train.values)
    if method=='CatBoost with Quantile Regression':
        quantiles = [0.025, 0.5, 0.975]
        for q in quantiles:
            if q == 0.025:
                parameters = {'loss_function': 'Quantile:alpha={:0.2f}'.format(q),
                          'num_boost_round': 5000}
                model_L = CatBoostRegressor(**parameters)
                model_L.fit(X_train.values, y_train.values,verbose=False)
            if q == 0.5:
                parameters = {'loss_function': 'Quantile:alpha={:0.2f}'.format(q),
                              'num_boost_round': 5000}
                model = CatBoostRegressor(**parameters)
                model.fit(X_train.values, y_train.values,verbose=False)
            if q == 0.975:
                parameters = {'loss_function': 'Quantile:alpha={:0.2f}'.format(q),
                          'num_boost_round': 5000}
                model_U = CatBoostRegressor(**parameters)
                model_U.fit(X_train.values, y_train.values,verbose=False)
    if method=='SVR':
        model = svm.SVR(C=235.16272869, kernel='rbf', gamma=2.79721385)
        #model = svm.SVR(C=255.76846667, kernel='rbf', gamma=2.5091838)
        #model = svm.SVR(C=254.02342761, kernel='rbf', gamma=2.81527426)
        model.fit(X_train.values, y_train.values)
    if method=='Random Forest':
        model = RandomForestRegressor(n_estimators=162, max_depth=23, min_samples_split=2,min_samples_leaf=1)
        #model = RandomForestRegressor(n_estimators=500, max_depth=44, min_samples_split=2, min_samples_leaf=1)
        #model = RandomForestRegressor(n_estimators=163, max_depth=28, min_samples_split=2, min_samples_leaf=1)
        model.fit(X_train.values, y_train.values)
    if method=='XGBoost':
        model = xgboost.XGBRegressor(n_estimators=int(403), max_depth=int(2), min_child_weight=1,
                                     gamma=0, subsample=8.67991090e-01, colsample_bytree=6.62013352e-01,
                                     reg_alpha=7.35862017e-01, reg_lambda=1, learning_rate=3.32361817e-01)
        #model = xgboost.XGBRegressor(n_estimators=int(357), max_depth=int(3), min_child_weight=1,
        #                             gamma=0, subsample=9.42853928e-01, colsample_bytree=6.42378569e-01,
        #                             reg_alpha=1.56964660e-01, reg_lambda=1, learning_rate=4.23657060e-01)
        #model = xgboost.XGBRegressor(n_estimators=int(387), max_depth=int(2), min_child_weight=5,
        #                             gamma=0, subsample=1, colsample_bytree=8.57030869e-01,
        #                             reg_alpha=3.16383362e-01, reg_lambda=1.34888649e-01, learning_rate=5.20778656e-01)
        model.fit(X_train.values, y_train.values)
    if method=='NGBoost':
        model = NGBRegressor(n_estimators=196, Dist=Normal, verbose=False,
                           learning_rate=0.31066895,verbose_eval=False)
        #model = NGBRegressor(n_estimators=191, Dist=Normal, verbose=False,
        #               learning_rate=0.320858,verbose_eval=False)
        #model = NGBRegressor(n_estimators=396, Dist=Normal, verbose=False,
        #                   learning_rate=0.25252533,verbose_eval=False)
        model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_train.values)
    y_pred_test = model.predict(X_test.values)
    y_pred_unc, y_pred_test_unc = [-1], [-1]
    # Probability model
    if method == 'Bayesian Ridge Regression':
        sig = model.sigma_
        y_pred = model.predict(X_train.values)
        y_pred_test = model.predict(X_test.values)
        y_pred_unc = np.sqrt(np.diagonal(np.dot(np.dot(X_train.values, sig),
                                                np.transpose(X_train.values))))
        y_pred_test_unc = np.sqrt(np.diagonal(np.dot(np.dot(X_test.values, sig),
                                                     np.transpose(X_test.values))))
    if method=='Quantile Regression' or method == 'CatBoost with Quantile Regression':
        y_pred_unc = (model_U.predict(X_train.values)-model_L.predict(X_train.values))/(2*1.96)
        y_pred_test_unc = (model_U.predict(X_test.values) - model_L.predict(X_test.values)) / (2*1.96)
    if method=='NGBoost':
        y_pred_unc = model.pred_dist(X_train.values).params['scale']
        y_pred_test_unc = model.pred_dist(X_test.values).params['scale']
    return y_pred, y_pred_test, y_pred_unc, y_pred_test_unc

def train_residual_learning_model(method,X_train,X_test,y_train):
    kernel = C(1.0) * Matern(length_scale=1.0) + WhiteKernel(noise_level=1.0) + C(1.0) * DotProduct(sigma_0=1.0)
    GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    if method=='Linear Regression with Residual Learning':
        model = LinearRegression()
        model.fit(X_train.values, y_train.values)
    if method=='Lasso Regression with Residual Learning':
        model = Lasso(0.00390625)
        model.fit(X_train.values, y_train.values)
    if method=='Bayesian Ridge Regression with Residual Learning':
        model = BayesianRidge(alpha_1=72.12096988,alpha_2=228.55120009,lambda_1=33.32712824,lambda_2=256)
        #model = BayesianRidge(alpha_1=96.04101504, alpha_2=256, lambda_1=35, lambda_2=256)
        #model = BayesianRidge(alpha_1=3.90625000e-03, alpha_2=2.26653768e+02, lambda_1=2.55582038e+01, lambda_2=256)
        model.fit(X_train.values, y_train.values)
    if method=='Quatile Regression with Residual learning':
        quantiles = [0.025, 0.5, 0.975]
        for q in quantiles:
            if q == 0.025:
                model_L = QuantileRegressor(quantile=q, alpha=0.001)
                model_L.fit(X_train.values, y_train.values)
            if q == 0.5:
                model = QuantileRegressor(quantile=q, alpha=0.001)
                model.fit(X_train.values, y_train.values)
            if q == 0.975:
                model_U = QuantileRegressor(quantile=q, alpha=0.001)
                model_U.fit(X_train.values, y_train.values)
                          
    res = y_train.values - model.predict(X_train.values)
    GPR.fit(X_train.values, res)
    y_res,std_res = GPR.predict(X_train.values,return_std=True)
    y_res_test,std_res_test = GPR.predict(X_test.values,return_std=True)
    y_pred = model.predict(X_train.values) + y_res
    y_pred_test = model.predict(X_test.values)+y_res_test

    y_pred_unc, y_pred_test_unc = [-1],[-1]
    if method == 'Bayesian Ridge Regression with Residual Learning':
      sig = model.sigma_
      y_pred_unc = np.sqrt(np.diagonal(np.dot(np.dot(X_train.values,sig),
                                          np.transpose(X_train.values)))
                         +std_res**2)
      y_pred_test_unc = np.sqrt(np.diagonal(np.dot(np.dot(X_test.values, sig),
                                                np.transpose(X_test.values)))
                             +std_res_test**2)
    if method == 'Quatile Regression with Residual learning':
        y_pred_unc = (model_U.predict(X_train.values)-model_L.predict(X_train.values))/(2*1.96)
        y_pred_test_unc = (model_U.predict(X_test.values) - model_L.predict(X_test.values)) / (2*1.96)
        
    
    return y_pred,y_pred_test,y_pred_unc,y_pred_test_unc

def coverage(y, yL, yH):
    return (100 / y.shape[0] * ((y > yL) & (y < yH)).sum())

def plot(method,y_train,y_test,y_pred,y_pred_test,n):
    RMSE = []
    MAE = []
    R2 = []
    RMSE_test = []
    MAE_test = []
    R2_test = []
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))
    mae_train = mean_absolute_error(y_train, y_pred)
    r2_train = r2_score(y_train, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    RMSE.append(rmse_train)
    MAE.append(mae_train)
    RMSE_test.append(rmse_test)
    MAE_test.append(mae_test)
    R2.append(r2_train)
    R2_test.append(r2_test)
    y_True = [y_train, y_test]
    y_Pre = [y_pred, y_pred_test]
    rmse = [rmse_train, rmse_test]
    mae = [mae_train, mae_test]
    r2 = [r2_train, r2_test]

    for j in np.arange(0, 2):
        unc = []
        for i in np.arange(0, len(y_Pre[j])):
            unc.append(abs(np.array(y_Pre[j])[i] - np.array(y_True[j])[i]))
        # Plot
        plt.figure()
        if j == 0:
            set = 'Train'
            y_T = y_train.values
            y_P = y_pred
        else:
            set = 'Test'
            y_T = y_test.values
            y_P = y_pred_test
        plt.plot(np.arange(1, len(y_T) + 1), sorted(y_P), label='predicted values', color='k',
                 linewidth=4)
        plt.scatter(np.arange(1, len(y_T) + 1), y_T[np.argsort(y_P)],
                    marker='o', color='darkblue',
                    label='observed values')
        plt.grid(True, which='major', linestyle='-',
                 linewidth='0.25', color='gray')
        plt.ylabel('log(Rupture Life) [hrs]')
        plt.xlabel('Data index in ascending order')
        plt.ylim(-2, 18)
        plt.xlim(0)
        plt.title(method +' ' + set)
        plt.legend()
        #plt.show()
        plt.savefig('./fig/predict/synall/'+method +set+str(n)+'_1.png')
        plt.close()

        plt.figure()
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(np.array(y_True[j]), y_Pre[j], c=unc, cmap=cm)
        plt.colorbar(sc)
        x_line = np.linspace(0.5, 13 + 0.5, 1000)
        y_line = x_line
        plt.plot(x_line, y_line, c='k')
        if j == 0:
            set = 'Train'
        else:
            set = 'Test'
        plt.title(method +' ' + set)
        plt.xlabel('Observed Data')
        plt.ylabel('Predicted Data')
        string_1 = 'RMSE = ' + '%f' % rmse[j]
        string_3 = 'MAE = ' + '%f' % mae[j]
        string_5 = 'R2 = ' + '%f' % r2[j]
        string = string_1 + '\n' + string_3 + '\n' + string_5
        plt.text(8, 1, string, fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='blue', lw=2, alpha=0.8))
        #plt.show()
        plt.savefig('./fig/predict/synall/' + method + set +str(n)+ '_2.png')
        plt.close()

    RMSE_train = np.mean(np.array(RMSE))
    RMSE_test_ = np.mean(np.array(RMSE_test))
    MAE_train = np.mean(np.array(MAE))
    MAE_test_ = np.mean(np.array(MAE_test))
    R2_train = np.mean(np.array(R2))
    R2_test_ = np.mean(np.array(R2_test))
    print(f'RMSE 的值为:{RMSE}')
    print(f'RMSE_train 的值为:{RMSE_train}')
    print(method)
    print(RMSE_train)
    print(RMSE_test_)
    print(MAE_train)
    print(MAE_test_)
    print(R2_train)
    print(R2_test_)
    return RMSE_train, RMSE_test_, MAE_train, MAE_test_, R2_train, R2_test_


def plot_unc(method,y_train,y_test,y_pred,y_pred_test,y_pred_unc, y_pred_test_unc,n):
    RMSE = []
    MAE = []
    R2 = []
    RMSE_test = []
    MAE_test = []
    R2_test = []
    Cover = []
    Cover_test = []
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred))
    mae_train = mean_absolute_error(y_train, y_pred)
    r2_train = r2_score(y_train, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    RMSE.append(rmse_train)#相当于赋值
    MAE.append(mae_train)
    RMSE_test.append(rmse_test)
    MAE_test.append(mae_test)
    R2.append(r2_train)
    R2_test.append(r2_test)
    y_True = [y_train, y_test]
    y_Pre = [y_pred, y_pred_test]
    rmse = [rmse_train, rmse_test]
    mae = [mae_train, mae_test]
    r2 = [r2_train, r2_test]

    for j in np.arange(0, 2):
        unc = []
        for i in np.arange(0, len(y_Pre[j])):
            unc.append(abs(np.array(y_Pre[j])[i] - np.array(y_True[j])[i]))
        # Plot
        plt.figure()
        if j == 0:
            set = 'Train'
            y_T = y_train.values
            y_P = y_pred
            y_U = y_pred + y_pred_unc * 1.96#95%的置信区间
            y_L = y_pred - y_pred_unc * 1.96
            Cover.append(coverage(y_T, y_L, y_U))
        else:
            set = 'Test'
            y_T = y_test.values
            y_P = y_pred_test
            y_U = y_pred_test + y_pred_test_unc * 1.96
            y_L = y_pred_test - y_pred_test_unc * 1.96
            Cover_test.append(coverage(y_T, y_L, y_U))
        plt.plot(np.arange(1, len(y_T) + 1), sorted(y_P), label='predicted values', color='k',
                 linewidth=4)
        plt.scatter(np.arange(1, len(y_T) + 1), y_T[np.argsort(y_P)],
                    marker='o', color='darkblue',
                    label='observed values')
        plt.fill_between(np.arange(1, len(y_T) + 1),
                         y_L[np.argsort(y_P)],
                         y_U[np.argsort(y_P)], alpha=0.3,
                         color='green',
                         label='95% confidence interval')
        plt.grid(True, which='major', linestyle='-',
                 linewidth='0.25', color='gray')
        plt.ylabel('log(Rupture Life) [hrs]')
        plt.xlabel('Data index in ascending order')
        plt.ylim(-2, 18)
        plt.xlim(0)
        plt.title(method +' '+ set)
        plt.legend()
        #plt.show()
        plt.savefig('./fig/predict/synall/' + method  + set + str(n)+'_1.png')
        plt.close()

        plt.figure()
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(np.array(y_True[j]), y_Pre[j], c=unc, cmap=cm)
        plt.colorbar(sc)
        x_line = np.linspace(0.5, 13 + 0.5, 1000)
        y_line = x_line
        plt.plot(x_line, y_line, c='k')
        if j == 0:
            set = 'Train'
        else:
            set = 'Test'
        plt.title(method +' ' + set)
        plt.xlabel('Observed Data')
        plt.ylabel('Predicted Data')
        string_1 = 'RMSE = ' + '%f' % rmse[j]
        string_3 = 'MAE = ' + '%f' % mae[j]
        string_5 = 'R2 = ' + '%f' % r2[j]
        string = string_1 + '\n' + string_3 + '\n' + string_5
        plt.text(8, 1, string, fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='blue', lw=2, alpha=0.8))
        #plt.show()
        plt.savefig('./fig/predict/synall/' + method  + set +str(n)+ '_2.png')
        plt.close()

        print(f"Coverage ({set}): {coverage(y_T, y_L, y_U)}")
        print(f"Upper coverage ({set}): {coverage(y_T, y_L, np.inf)}")
        print(f"Lower coverage ({set}): {coverage(y_T, -np.inf, y_U)}")

    print(method)
    Coverage_train = np.mean(np.array(Cover))
    Coverage_test = np.mean(np.array(Cover_test))
    print(Coverage_train)
    print(Coverage_test)
    RMSE_train = np.mean(np.array(RMSE))
    RMSE_test_ = np.mean(np.array(RMSE_test))
    MAE_train = np.mean(np.array(MAE))
    MAE_test_ = np.mean(np.array(MAE_test))
    R2_train = np.mean(np.array(R2))
    R2_test_ = np.mean(np.array(R2_test))
    print(RMSE_train)
    print(RMSE_test_)
    print(MAE_train)
    print(MAE_test_)
    print(R2_train)
    print(R2_test_)
    return RMSE_train,RMSE_test_,MAE_train,MAE_test_,R2_train,R2_test_



if __name__ == '__main__':
    
    # esc = 600
    data_dir = "./data/real data/data_28fea_logtar.csv"
    fake_dir = "./data/fake data/fake_data_28_685.csv"
    X, y, data, X_f, y_f, data_fake = load_data(data_dir,fake_dir)

    #methods = ['Linear Regression','Lasso Regression','Bayesian Ridge Regression','Quantile Regression',
    #           'Quantile Regression with CatBoost','SVR','Random Forest','XGBoost','NGBoost']
    methods = ['CatBoost with Quantile Regression','SVR','Random Forest','XGBoost','NGBoost']
    methods_2 = ['Linear Regression with Residual Learning','Lasso Regression with Residual Learning',
                 'Bayesian Ridge Regression with Residual Learning']

    metrics = pd.DataFrame(columns=('Method','RMSE_train','MAE_train','R2_train','RMSE_test','MAE_test','R2_test'))

    for method in methods:
      RMSE_n = []
      MAE_n = []
      R2_n = []
      RMSE_test_n = []
      MAE_test_n = []
      R2_test_n = []
      for n in np.arange(0, 5, 1):#做五次平行试验
         X_train, X_test, y_train, y_test = train_test(data, 0.7)
         # X_f_1 = X_f.iloc[:esc,:] 
         # y_f_1 = y_f[:esc]
         X_train = pd.concat([X_train,X_f])
         y_train = pd.concat([y_train,y_f])
         y_pred, y_pred_test, y_pred_unc, y_pred_test_unc = \
             train_simple_model(method,X_train,X_test,y_train)
         if y_pred_unc[0]!=-1:
             RMSE_train,RMSE_test_,MAE_train,MAE_test_,R2_train,R2_test_ = \
                 plot_unc(method,y_train,y_test,y_pred,y_pred_test,y_pred_unc, y_pred_test_unc,n)
         else: RMSE_train,RMSE_test_,MAE_train,MAE_test_,R2_train,R2_test_ = \
             plot(method,y_train,y_test,y_pred,y_pred_test,n)
         RMSE_n.append(RMSE_train)
         MAE_n.append(MAE_train)
         R2_n.append(R2_train)
         RMSE_test_n.append(RMSE_test_)
         MAE_test_n.append(MAE_test_)
         R2_test_n.append(R2_test_)
      series = pd.Series({"Method":method,'RMSE_train':np.min(np.array(RMSE_n)),'MAE_train':np.min(np.array(MAE_n)),
                          'R2_train':np.max(np.array(R2_n)),'RMSE_test':np.min(np.array(RMSE_test_n)),
                          'MAE_test':np.min(np.array(MAE_test_n)),'R2_test':np.max(np.array(R2_test_n))},name=method)
      metrics = metrics.append(series,ignore_index=True)


    for method in methods_2:
      RMSE_n = []
      MAE_n = []
      R2_n = []
      RMSE_test_n = []
      MAE_test_n = []
      R2_test_n = []
      for n in np.arange(0, 5, 1):
         X_train, X_test, y_train, y_test = train_test(data, 0.7)
         # X_f_1 = X_f.iloc[:esc,:] 
         # y_f_1 = y_f[:esc]
         X_train = pd.concat([X_train,X_f])
         y_train = pd.concat([y_train,y_f])
         y_pred, y_pred_test, y_pred_unc, y_pred_test_unc = \
             train_residual_learning_model(method,X_train,X_test,y_train)
         if y_pred_unc[0]!=-1:
             RMSE_train,RMSE_test_,MAE_train,MAE_test_,R2_train,R2_test_ =\
                 plot_unc(method,y_train,y_test,y_pred,y_pred_test,y_pred_unc, y_pred_test_unc,n)
         else: RMSE_train,RMSE_test_,MAE_train,MAE_test_,R2_train,R2_test_ =\
             plot(method,y_train,y_test,y_pred,y_pred_test,n)
         RMSE_n.append(RMSE_train)
         MAE_n.append(MAE_train)
         R2_n.append(R2_train)
         RMSE_test_n.append(RMSE_test_)
         MAE_test_n.append(MAE_test_)
         R2_test_n.append(R2_test_)
         
      series = pd.Series({"Method":method,'RMSE_train':np.min(np.array(RMSE_n)),'MAE_train':np.min(np.array(MAE_n)),
                          'R2_train':np.max(np.array(R2_n)),'RMSE_test':np.min(np.array(RMSE_test_n)),
                          'MAE_test':np.min(np.array(MAE_test_n)),'R2_test':np.max(np.array(R2_test_n))},name=method)
      metrics = metrics.append(series,ignore_index=True)

    metrics.to_excel('./Results/predict/results_synall.xlsx')