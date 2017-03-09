import sys
import os
sys.path.append(os.getcwd())
import json
import requests
import pandas as pd
import pdb
import time
from api_login import username, password #Provide file with your own credentials to access the intrinio web-API, otherwise use the file 'large_dataset_nodates.csv'

prices_url='https://api.intrinio.com/prices'
hist_url = 'https://api.intrinio.com/historical_data'
pattern='%Y-%m-%d'


def check_status(status_code):
	'''Check HTTP status codes '''
	status=status_code
	if status == 401:
		print 'Error: API Authorization is not authenticated'
	elif status == 403:
		print 'Error: Not subscribed to the requested data'
	elif status == 404:
		print 'Error: End point not found'
	elif status == 429:
		print "Error: Too many requests"
	elif status == 500:
		print "Internal server error"
	elif status == 503:
		print "Service unavailable"
	return


def get_prices(symbol='', start_date='', end_date='today'):
	''' Retrieve stock price data for given symbol and start, end dates'''

	ticker = symbol
	params = {'ticker':ticker, 'start_date':start_date, 'end_date':end_date} #identifier = ticker, item, sequence, start_date, end_date, frequency
	r = requests.get(prices_url, params=params, auth=(username,password))
	check_status(r.status_code)
	data = json.loads(r.text)

	#5th key of raw data = data. Data keys = 'ex_dividend', 'volume', 'adj_low', 'adj_open', 'adj_close', 'high', 'adj_volume', 'low', 'date', 'close', 'open', 'adj_high', 'split_ratio'
	d={}
	idx={}
	items = ['volume','open','high','low','adj_close']
	for item in items:
		d[item]=[day[item] for day in data['data']]
	idx['epoch']=[int(time.mktime(time.strptime(day['date'],pattern))) for day in data['data'] ]
	df=pd.DataFrame(data=d,index=idx['epoch'])
	df.rename(columns={'volume':'volume_'+symbol, 'open':'open_'+symbol, 'high':'high_'+symbol, 'low':'low_'+symbol, 'adj_close':'adj_close_'+symbol, 'date':'date_'+symbol},inplace=True)
	print 'Get stock prices request for '+symbol+' finished'
	return df


def get_indices(identifier='$NDX', start_date='', end_date='today'):
	''' Retrieve market index data for given identifier and start, end dates'''

	param={'identifier':identifier, 'item':'close_price', 'start_date':start_date, 'end_date':end_date}
	r = requests.get(hist_url,params = param, auth=(username,password))
	check_status(r.status_code)
	data = json.loads(r.text)
	d={}
	idx={}
	items=['value']
	for item in items:
		d[item]=[day[item] for day in data['data']]
	idx['epoch']=[int(time.mktime(time.strptime(day['date'], pattern))) for day in data['data'] ]
	df=pd.DataFrame(data=d,index=idx['epoch'])
	df.rename(columns={'value':'close_price_'+identifier,'date':'date_'+identifier},inplace=True)
	print 'Get market index request for '+identifier+' finished'
	return df