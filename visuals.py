import sys
import os
sys.path.append(os.getcwd())

from utility import *
from predictor import *
import pandas as pd
import numpy as np
import pdb
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import make_scorer
import sklearn.learning_curve as curves
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


def data_plotter(data):
    '''Produces a figure of raw stock data over time '''
    fig, (ax, ax1) = plt.subplots(2,1, figsize=(8,6), dpi=50)

    symbols = ['BAC', 'QCOM', 'AAPL', 'AMZN']
    indices = ['$NDX', '$SPX']
    for symbol in symbols:
        selected_col = [column for column in data.columns if symbol in column and 'close' in column]     
        max_scale = data[selected_col].max()[0]
        x_axis = data[selected_col].index.values.tolist()    
        ax.plot(x_axis, data[selected_col]/max_scale, linewidth=1, label = symbol, linestyle = '-')
        ax.legend(bbox_to_anchor=(-0.1, 1.05))


    for index in indices:
        selected_col = [column for column in data.columns if index in column and 'close' in column]
        x_axis = data[selected_col].index.values.tolist()    
        max_scale = data[selected_col].max()[0]
        ax1.plot(x_axis, data[selected_col]/max_scale, linewidth=1, label = index[1:], linestyle = '-')    
        ax1.legend(bbox_to_anchor=(-0.1, 1.05))

    plt.subplots_adjust(hspace = 0.4)
    start, end = ax.get_xlim()
    ticks = np.arange(start, end, 31556926)
    labels = [datetime.datetime.strptime(convert_from_epoch(i), '%Y-%m-%d').strftime('%Y-%m') for i in ticks]
    ax.xaxis.set_ticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Adjusted Closing Price (Normalized)')
    ax.set_xlabel('Date')
    ax.set_title('Adjusted Close Prices', fontsize = 15)


    ax1.xaxis.set_ticks(ticks)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Close Index (Normalized)')
    ax1.set_xlabel('Date')    
    ax1.set_title('Market Indices', fontsize = 15)
    return fig


def feature_plotter(features):
    '''Produces a figure of all features over time for a given stock symbol '''

    #1 year = 31556926 seconds
    fig = plt.figure(figsize = (16, 12), dpi = 100)
    vol_cols = [col for col in features.columns if 'volatility' in col]

    mom_cols = [col for col in features.columns if 'momentum' in col]

    boll_cols = [col for col in features.columns if 'boll' in col]

    symbols = np.unique([i[:3] for i in boll_cols])

    ax0 = plt.subplot2grid((2,2),(0,0))
    ax1 = plt.subplot2grid((2,2),(0,1))
    ax2 = plt.subplot2grid((2,2),(1,0), colspan = 2)

    for vol in vol_cols:     
        vol_axis = features[vol][::-1].index.values.tolist()
        ax0.plot(vol_axis, features[vol][::-1], 'm')
        start, end = ax0.get_xlim()
        ticks  = np.arange(start, end, 31556926)
        labels = [datetime.datetime.strptime(convert_from_epoch(i), '%Y-%m-%d').strftime('%Y-%m') for i in ticks]
        ax0.set_xticklabels(labels)
        ax0.xaxis.set_ticks(ticks)
        ax0.set_ylabel("Stock Closing Price Volatility")
        ax0.set_xlabel('Date')
        ax0.set_title(symbols[0]+' ACP Volatility', fontsize = 15)

    for mom in mom_cols:
        mom_axis = features[mom][::-1].index.values.tolist()
        ax1.plot(mom_axis, features[mom][::-1], 'y')
        ax1.set_xticklabels(labels)
        ax1.xaxis.set_ticks(ticks)
        ax1.set_ylabel('Stock Closing Price Momentum')
        ax1.set_xlabel('Date')
        ax1.set_title(symbols[0]+' ACP Momentum', fontsize = 15)

    for symbol in symbols:
        columns = [col for col in boll_cols if symbol in col]
        for col in columns:
            boll_axis = features[col][::-1].index.values.tolist()
            ax2.plot(boll_axis, features[col][::-1])
            ax2.set_xticklabels(labels)
            ax2.xaxis.set_ticks(ticks)
            ax2.set_ylabel('Stock Closing Price (Bollinger Bands)')
            ax2.set_title(symbol+' Bollinger Bands', fontsize = 15)
            ax2.set_xlabel('Date')
    fig.suptitle('Stock Prediction Features', fontsize = 20)
    fig.savefig('img/'+symbol+'feature_plots.png')
    return fig


def fold_plotter(train, test, pred):
    '''Produces a figure displaying training, test, and prediction folds '''

    num_folds = len(train)
    cols = 3
    #rows = (num_folds//cols)+(num_folds%cols)
    fig, ax = plt.subplots(1, cols, figsize = (12, 4), dpi = 100)
    preds=[0]*len(pred)
    preds[:]=pred[:]
    plot_folds = [0,num_folds/2,num_folds-1]
    k=0
    for i in range(num_folds):
        if i>0:
            preds[i] = preds[i].merge(preds[i-1], left_index = True, right_index = True, how = 'outer')         
        train_axis = train[i].index.values.tolist()
        test_axis = test[i].index.values.tolist()
        pred_axis = preds[i].index.values.tolist()

        if i in plot_folds:
            ax[k].plot(train_axis, train[i], 'b', test_axis, test[i], 'r', pred_axis, preds[i], 'g')

            start, end = ax[k].get_xlim()

            ticks  = np.arange(start, end, (end-start)/4 )
            labels = [datetime.datetime.strptime(convert_from_epoch(tick), '%Y-%m-%d').strftime('%Y-%m') for tick in ticks]

            ax[k].set_xticklabels(labels)
            ax[k].xaxis.set_ticks(ticks)
            ax[k].set_ylabel('Adjusted Close Price')
            ax[k].set_xlabel('Date')
            if i==0:
                ax[k].legend(['Training Data', 'Test Data', 'Predictions'], bbox_to_anchor=(-0.12, 1.05))
            k+=1

    fig.savefig('img/'+'fold_plots.png')
    return fig, ax


def model_complexity(features, targets, cv_data):
    '''Produces a figure displaying the model complexity curve--model score performance as a function of parameter tuning '''

    from predictor import performance_metric
    params = ['base_estimator', 'n_estimators', 'learning_rate']
    params_range = [ [DecisionTreeRegressor(max_depth = 2), DecisionTreeRegressor(max_depth = 4), DecisionTreeRegressor(max_depth = 8), DecisionTreeRegressor(max_depth = 16), DecisionTreeRegressor(max_depth = 32)], \
     np.arange(25, 250, 25), np.arange(0.5, 3, 0.5)]
    pcurve = zip(params,params_range)
    scoring_fnc = make_scorer(performance_metric, greater_is_better = False)
    k=0
    com_fig, com_ax = plt.subplots(1,3, figsize = (12, 4), dpi = 50)
    for pname, prange in pcurve:
        train_scores, test_scores = curves.validation_curve(AdaBoostRegressor(), features, targets, param_name = pname, param_range = prange, cv = cv_data, scoring = scoring_fnc)
        train_mean = np.mean(abs(train_scores), axis = 1)
        test_mean = np.mean(abs(test_scores), axis = 1)
        train_std = np.std(abs(train_scores), axis = 1)
        test_std = np.std(abs(test_scores), axis = 1)
        if k==0:
            prange = [2,4,8,16,32]
        com_ax[k].plot(prange, train_mean, 'o-', color = 'r', label = 'Training Score')
        com_ax[k].plot(prange, test_mean, 'o-', color = 'g', label = 'Validation Score')
        com_ax[k].set_ylim((0,14))
        com_ax[k].fill_between(prange, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
        com_ax[k].fill_between(prange, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

        if k==0:
            com_ax[k].set_xlabel('{} max_depth'.format(pname))
            com_ax[k].legend(['Training Score', 'Testing Score' ], bbox_to_anchor=(-.16, 1.05))
        else:
            com_ax[k].set_xlabel(pname)
        com_ax[k].set_ylabel('Adjusted Close Price Relative Difference (%)')
        k+=1
    com_fig.savefig('img/'+'col_plots.png')
    return com_fig, com_ax


def model_complexity_partial(features, targets, cv_data):
    '''Produces a figure displaying the model complexity curve--model score performance as a function of parameter tuning. Plot only latter half of fold data'''

    from predictor import performance_metric
    params = ['base_estimator', 'n_estimators', 'learning_rate']
    params_range = [ [DecisionTreeRegressor(max_depth = 2), DecisionTreeRegressor(max_depth = 4), DecisionTreeRegressor(max_depth = 8), DecisionTreeRegressor(max_depth = 16), DecisionTreeRegressor(max_depth = 32)], \
     np.arange(25, 250, 25), np.arange(0.5, 3, 0.5)]
    pcurve = zip(params,params_range)
    
    scoring_fnc = make_scorer(performance_metric, greater_is_better = False)

    k=0
    com_fig, com_ax = plt.subplots(1,3, figsize = (12, 4), dpi = 50)

    for pname, prange in pcurve:
        train_scores, test_scores = curves.validation_curve(AdaBoostRegressor(), features, targets, param_name = pname, param_range = prange, cv = cv_data, scoring = scoring_fnc)

        train_mean = np.mean(abs(train_scores[:,:47]), axis = 1)
        test_mean = np.mean(abs(test_scores[:,:47]), axis = 1)
        train_std = np.std(abs(train_scores[:,:47]), axis = 1)
        test_std = np.std(abs(test_scores[:,:47]), axis = 1)

        if k==0:
            prange = [2,4,8,16,32]
        com_ax[k].plot(prange, train_mean, 'o-', color = 'r', label = 'Training Score')
        com_ax[k].plot(prange, test_mean, 'o-', color = 'g', label = 'Validation Score')
        com_ax[k].set_ylim((0,14))
        com_ax[k].fill_between(prange, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
        com_ax[k].fill_between(prange, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

        if k==0:
            com_ax[k].set_xlabel('{} max_depth'.format(pname))
            com_ax[k].legend(['Training Score', 'Testing Score' ], bbox_to_anchor=(-.16, 1.05))
        else:
            com_ax[k].set_xlabel(pname)
        com_ax[k].set_ylabel('Adjusted Close Price Relative Difference (%)')
        k+=1
    com_fig.savefig('img/'+'col_plots.png')
    return com_fig, com_ax


def learning_curve(test_preds, test_folds, train_preds, train_folds):
    '''Produces a figure displaying the learning curve--model score performance as a function of training samples.'''

    from predictor import performance_metric
    test_scores = [performance_metric(test_folds[i].values, test_preds[i].values.ravel()) for i in range(len(test_folds))]
    train_scores = [performance_metric(train_folds[i].values, train_preds[i].values.ravel()) for i in range(len(train_folds))]

    sizes = [len(i) for i in train_folds]
    fig, ax = plt.subplots(1,1, figsize = (12, 3), dpi = 100)
    ax.plot(sizes, train_scores, 'o-', color = 'r', label = 'Training Set')
    ax.plot(sizes, test_scores, 'o-', color = 'g', label = 'Testing Set')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Adjusted Close Price Relative Difference (%)')
    ax.legend(['Training Score', 'Testing Score' ], bbox_to_anchor=(-.07, 1.05))
    fig.savefig('img/'+'learningcurve.png')
    return fig, ax