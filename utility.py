import sys
import os
sys.path.append(os.getcwd())
from time import gmtime, strftime
import datetime
import time
import pandas as pd
import numpy as np
import pdb
from intrinioapi import *


def add_days(date_str,days):
	'''Computes the date string X which is the given date Y plus the number of days after Y.'''
	start_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
	end_date = start_date + datetime.timedelta(days=days)
	end_date = end_date.strftime('%Y-%m-%d')
	return int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))


def find_nearest_date(date,data):
	'''Returns the date (in epoch) nearest the input date'''
	date = convert_to_epoch(date)
	nearest_date = min(data.index.values, key=lambda x: abs(x-date))
	return nearest_date


def convert_from_epoch(epoch_time):
	'''Converts to date string from epoch'''
	return time.strftime('%Y-%m-%d', time.localtime(epoch_time))


def convert_to_epoch(date_str):
    '''Converts to epoch from date string'''
    return int(time.mktime(time.strptime(date_str,'%Y-%m-%d')))


def get_data(file=True, stocks=['BAC','AAPL','AMZN','QCOM'], indices=['$NDX','$SPX'], start_date = '2010-01-01', end_date = 'today'):
    '''Returns structured dataset either from file or from intrinio web-API'''

    if file:
        data = from_file()
    else:
        data = build_dataset(stocks, indices, start_date = '2010-01-01', end_date = 'today')
    return data


def build_dataset(stocks=['BAC', 'QCOM', 'AAPL', 'AMZN'],indices=['$NDX','$SPX'], start_date = '2010-01-01', end_date = 'today'):
	'''Builds structured dataset from the intrinio web-API '''

	index_list = [0]*len(indices)
	stock_list = [0]*len(stocks)
	for i,index in enumerate(indices):
	    index_list[i] = get_indices(index, start_date, end_date)

	for j,stock in enumerate(stocks):
	    stock_list[j] = get_prices(stock, start_date, end_date)
	combined = stock_list+index_list
	data = reduce(lambda left, right : pd.merge( left, right, left_index=True, right_index=True, how='outer'),combined)
	data.fillna(method='ffill',inplace=True)
	data.fillna(method='bfill',inplace=True)
	data=data[::-1]
	return data


def from_file():
    '''Load dataset from file'''
    data = pd.DataFrame.from_csv('large_dataset_nodates.csv')
    print 'Imported data from file'
    return data
