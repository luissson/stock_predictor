import sys
import os
sys.path.append(os.getcwd())
from IPython.display import display
from pred_gui import launch_gui
from intrinioapi import get_prices,get_indices
from utility import *
from visuals import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from time import gmtime, strftime
import datetime
import time
import pdb
pd.options.mode.chained_assignment = None



def main(data = pd.DataFrame(), symbol='BAC', predict_date=7, end_date = '2016-12-31', cv = False):
    '''Calls necessary functions to train model and produce predictions, as well as output analysis depending on cross-validation arg '''

    feature_data, target_data, pred_features, actual_price, test_data, fig = data_processing(data, symbol, predict_date, end_date, 5)
    last_train_price = target_data.iloc[0]
    num_folds = 13 #Number of folds for cross-validation

    cv_data = time_series_cv(len(feature_data), num_folds)

    if cv==True:
        #Produce iterable for cross-validation e.g. [(split1_train_idxs, split1_test_idxs), (split2_train_idxs, split2_test_idxs), (split3_train_idxs, split3_test_idxs),...]   
        #cv_data = time_series_cv(len(feature_data), num_folds)

        #Partition feature and target datasets into corresponding folds, return best regressor parameters (if GridSearchCV is enabled, takes several minutes to run)
        regr_params, train_preds, train_folds, test_preds, test_folds, feature_folds = cross_validation(feature_data, target_data, cv_data)
        
        #Produce complexity curve figure
        com_fig, com_ax = model_complexity(feature_data, target_data, cv_data)

        #Produce learning curve figure
        learn_fig, learn_ax = learning_curve(test_preds, test_folds, train_preds, train_folds )

        #Produce set of folds figure
        fold_fig, fold_ax = fold_plotter(train_folds, test_folds, test_preds)

        #Predict stock price
        pred_price = fit_pred_price(feature_data, target_data, pred_features, regr_params)
        svm_params = {'C':4096 , 'gamma': 2**(-22)}        
        svm_pred = svm_pred_price(feature_data, target_data, pred_features, svm_params)
        print 'Prediction Relative Difference: {}'.format(performance_metric(actual_price['adj_close_'+symbol], pred_price))
        plt.show() 



    else:
        #Use previous best regressor params to predict stock price
        regr_params = {'base_estimator': DecisionTreeRegressor(max_depth=16), 'n_estimators':150, 'learning_rate':2, 'random_state':99}
        pred_price = fit_pred_price(feature_data, target_data, pred_features, regr_params)
        svm_params = {'C':4096 , 'gamma': 2**(-22)}
        svm_pred = svm_pred_price(feature_data, target_data, pred_features, svm_params)
        print 'Prediction Relative Difference: {}'.format(performance_metric(actual_price, pred_price))



def data_processing(data, symbol, predict_date, end_date, avg_num):
    '''Takes in raw data from the intrinio API and prepares feature and target columns for the model '''

    #Compute the difference in days between the prediction day and the last training day
    last_train_date = find_nearest_date(end_date, data)

    end_index = np.where(data.index.values == last_train_date)[0][0]

    split_index = np.where(data.index.values == last_train_date)[0][0]

    #Split dataset into training and "future" datasets
    select_cols = [column for column in data.columns if 'close' in column]    
    training_set = data[select_cols][end_index:]
    unseen_set = data[select_cols][:end_index]
    predict_index = len(unseen_set) - predict_date
    #Create Features
    for column in training_set.columns:
        add_features(training_set,training_set[column], avg_num, column[-3:])

    #Shift training set by the difference in days between the prediction day and the last training day (taken to be current day)
    training_set['target'] = training_set['adj_close_'+symbol].shift(predict_date)

    #Split training dataset into feature and target datasets
    exclude = ['close', 'target']
    features = [column for column in training_set.columns if all( x not in column for x in exclude )]
    feature_data = training_set[features]
    target_data = training_set['target']
    actual_price = unseen_set.iloc[predict_index]
    special_cols = [i for i in feature_data.columns if symbol[-3:] in i]
    pred_features = feature_data.iloc[0]

    feature_data.drop(training_set.index[:predict_date], inplace = True) #Remove most current trading days that have no corresponding target
    feature_data.drop(training_set.index[-avg_num:], inplace = True) #Remove oldest trading days that have no corresponding feature samples.

    target_data.drop(training_set.index[:predict_date], inplace = True) #Remove most current trading days to keep consistent sizing
    target_data.drop(training_set.index[-avg_num:], inplace = True) #Remove oldest trading days to keep consistent sizing

    fig = feature_plotter(feature_data[special_cols])

    return feature_data, target_data, pred_features, actual_price, unseen_set, fig


def data_processing_nofig(data, symbol, predict_date, end_date, avg_num):
    ''' Same as data_processing but remove the call to create a figure for jupyter notebook display'''

    #Compute the difference in days between the prediction day and the last training day
    last_train_date = find_nearest_date(end_date, data)

    end_index = np.where(data.index.values == last_train_date)[0][0]

    split_index = np.where(data.index.values == last_train_date)[0][0]

    #Split dataset into training and "future" datasets
    select_cols = [column for column in data.columns if 'close' in column]    
    training_set = data[select_cols][end_index:]
    unseen_set = data[select_cols][:end_index]
    predict_index = len(unseen_set) - predict_date
    #Create Features
    for column in training_set.columns:
        add_features(training_set,training_set[column], avg_num, column[-3:])

    #Shift training set by the difference in days between the prediction day and the last training day (taken to be current day)
    training_set['target'] = training_set['adj_close_'+symbol].shift(predict_date)

    #Split training dataset into feature and target datasets
    exclude = ['close', 'target']
    features = [column for column in training_set.columns if all( x not in column for x in exclude )]
    feature_data = training_set[features]
    target_data = training_set['target']
    actual_price = unseen_set.iloc[predict_index]

    pred_features = feature_data.iloc[0]

    feature_data.drop(training_set.index[:predict_date], inplace = True) #Remove most current trading days that have no corresponding target
    feature_data.drop(training_set.index[-avg_num:], inplace = True) #Remove oldest trading days that have no corresponding feature samples.

    target_data.drop(training_set.index[:predict_date], inplace = True) #Remove most current trading days to keep consistent sizing
    target_data.drop(training_set.index[-avg_num:], inplace = True) #Remove oldest trading days to keep consistent sizing

    return feature_data, target_data, pred_features, actual_price, unseen_set


def add_features(features,target,days_avg,id):
    '''Calls functions that compute features and places the resulting series in a pandas Dataframe '''
    features[id+'_volatility'] = feature_volatility(target, days_avg)
    features[id+'_momentum'] = feature_momentum(target, days_avg)
    features[id+'_upper_boll'], features[id+'_lower_boll'], features[id+'_mid_boll'] = bollinger_bands(target, days_avg)


def bollinger_bands(value_list, num_days):
    '''Computes Bollinger bands, returning middle, upper, and lower bands '''
    mid_band = value_list[::-1].rolling(num_days).mean()
    rolling_std = value_list[::-1].rolling(num_days).std()
    upper_band = mid_band + num_days*rolling_std
    lower_band = mid_band - num_days*rolling_std
    return mid_band, upper_band, lower_band


def feature_momentum(value_list, num_days):
    '''Computes the feature momentum'''
    feature_mom = ((value_list-value_list.shift(-1))>0).astype(float)
    feature_mom = feature_mom.replace(0.0, -1.0)
    feature_momentum = feature_mom[::-1].rolling(num_days).mean()
    return feature_momentum[::-1]


def feature_volatility(value_list, num_days):
    '''Computes the feature volatility'''
    pchange = 100*(value_list-value_list.shift(-1))/value_list.shift(-1)
    volatility = pchange[::-1].rolling(num_days).mean()
    return volatility[::-1]


def cross_validation(feature_data, target_data, cv_data):
    '''Uses cross-validation iterable to make predictions on each fold. Finds best regressor params using GridSearchCV if uncommented (slow) '''
    #regr_params = fit_model(feature_data, target_data, cv_data)
    regr_params = {'base_estimator':DecisionTreeRegressor(max_depth = 16), 'n_estimators':150, 'learning_rate':2, 'random_state' : 99}
    train_preds, train_folds, test_preds, test_folds, feature_folds = predict_on_folds(feature_data, target_data, cv_data, regr_params)
    return regr_params, train_preds, train_folds, test_preds, test_folds, feature_folds


def time_series_cv(feature_length, num_folds):
    ''' Produces scikit-learn friendly iterable for time-series cross-validation in the following form: [(split1_train_idxs, split1_test_idxs), (split2_train_idxs, split2_test_idxs), (split3_train_idxs, split3_test_idxs),...]'''
    k = int(np.floor(float(feature_length) / num_folds))
    indices_list = [0]*len(range(2,num_folds+1))
    for i in range(2, num_folds + 1):
        split = float(i-1)/i
        train_size = k*i
        index = int(np.floor(train_size * split))
        train_idx = np.arange(index)
        test_idx = np.arange((index+1),(k*i))
        indices_list[i-2]=(train_idx,test_idx)
    return indices_list


def predict_on_folds(xtrain, ytrain, index_list, params):
    '''Builds lists of pandas Dataframes containing training, test, and prediction folds'''
    num_folds = len(index_list)
    train_preds = [0]*(num_folds)
    preds = [0]*(num_folds)
    train_folds = [0]*(num_folds)
    test_folds = [0]*(num_folds)
    feature_folds = [0]*(num_folds)
    xtrain = xtrain[::-1]
    ytrain = ytrain[::-1]
    k=0
    for i,j in index_list:
        if k>0:
            del regr, pred, preds_df, train_pred, train_preds_df
        xtrain_fold = xtrain.iloc[i]
        ytrain_fold = ytrain.iloc[i]
        xtest_fold = xtrain.iloc[j]
        ytest_fold = ytrain.iloc[j]

        regr = AdaBoostRegressor(**params)
        regr.fit(xtrain_fold, ytrain_fold)
        train_pred = regr.predict(xtrain_fold)
        pred = regr.predict(xtest_fold)
        preds_df = pd.DataFrame(data = pred, index = xtest_fold.index.values)

        train_preds_df = pd.DataFrame(data = train_pred, index = xtrain_fold.index.values)

        feature_folds[k] = xtrain_fold     
        preds[k] = preds_df
        train_preds[k] = train_preds_df
        train_folds[k] = ytrain_fold
        test_folds[k] = ytest_fold
        k+=1
    return train_preds, train_folds, preds, test_folds, feature_folds


def fit_model(features, targets, cv_data = None):
    '''Uses GridSearchCV and a dictionary of params to find the best set of params given a scoring function'''
    regr = AdaBoostRegressor(random_state = 99)
    params = {'base_estimator':[DecisionTreeRegressor(max_depth = 8), DecisionTreeRegressor(max_depth = 16), DecisionTreeRegressor(max_depth = 32)] , 'n_estimators':np.arange(100, 200, 50), 'learning_rate':np.arange(1, 3, 1)}
    scoring_fnc = make_scorer(performance_metric, greater_is_better = False)
    grid = GridSearchCV(regr, params, cv = cv_data, scoring = scoring_fnc)
    grid.fit(features,targets.values.ravel())
    return grid.best_params_


def fit_pred_price(features, targets, test_features, params):
    '''Fit regressor with all but the last training date, predict using last date. '''
    if len(params) == 0:
        params = {'base_estimator':DecisionTreeRegressor(max_depth = 16), 'n_estimators':150, 'learning_rate':2,'random_state':99}
    regr = AdaBoostRegressor(**params)
    regr.fit(features,targets)

    pred = regr.predict(test_features.reshape(1,-1))
    return pred[0]


def svm_pred_price(features, targets, test_features, params):
    scoring_fnc = make_scorer(performance_metric, greater_is_better = False)
    #C_range = np.logspace(10, 12, 3, base = 2)
    #gamma_range = np.logspace(-30, -10, 6, base = 2)
    #param_grid = dict(gamma=gamma_range, C=C_range)   
    #grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv_data, scoring = scoring_fnc)
    #grid.fit(features, targets)
    #svm_params = grid.best_params_

    regr_svm = SVR(**params)
    regr_svm.fit(features,targets)
    pred = regr_svm.predict(test_features.reshape(1,-1))
    return pred[0]


def performance_metric(ytest, ypred):
    '''Computes relative difference given two numbers, or the mean relative difference given two lists'''
    try:
        int(ytest)
        score = 100*abs(ytest-ypred)/max(ytest,ypred)
    except TypeError:
        score=np.mean(100.*abs(ytest-ypred)/np.mean((ytest,ypred)))
    return score 


if __name__ == '__main__':

    end_date, symbols, intervals = launch_gui()

    #Bypass GUI for quick-debugging of main code
    #end_date = '2015-01-01'
    #symbols = 'BAC'
    #intervals = '5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160'

    try:
        predict_days = [int(intervals)]
    except ValueError:
        predict_days = map(int,intervals.strip().split(','))

    data = get_data(file = True , end_date = '2017-02-06')
    data_plotter(data)

    symbols = symbols.split(',')
    symbols = [symbol.strip() for symbol in symbols]

    for i in range(len(symbols)):
        for j in range(len(predict_days)):
            main(data, symbols[i], predict_days[j], end_date, True) #main args: data, symbol, predict_date, end_date, cv; where cv = cross-validation bool 



