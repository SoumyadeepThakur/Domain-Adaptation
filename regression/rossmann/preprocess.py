import pandas as pd
import numpy as np
import argparse
from pandas import datetime

def make_split(store_type, hist, splits):

	stores = {'a': 3, 'b': 259, 'c': 1, 'd': 13}
	store = stores[store_type]
	df = pd.read_csv('train.csv', parse_dates=True, index_col='Date')
	df = df.sort_index(ascending=True)
	df = df[(df.Store==store)]
	df['Month'] = df.index.month
	df['Day'] = df.index.day
	df['time'] = pd.Series(np.arange(df.shape[0], dtype=np.int32), index=df.index)
	df = df[['Sales', 'Month', 'Day', 'Open', 'Promo', 'time']]
	print(df.index)
	print(df)

	times = df['time'].values
	sales = df['Sales'].values
	month = df['Month'].values
	day = df['Day'].values
	op = df['Open'].values
	promo = df['Promo'].values


	min_sales = np.min(sales)
	max_sales = np.max(sales)
	print('Max and min sales: (%d, %d)' % (max_sales,min_sales))
	sales = 100 * (sales - min_sales)/(max_sales - min_sales)
	X = list()
	Y = list()
	for i in range(sales.shape[0] - hist):

		X.append(np.hstack([sales[i:i+hist], month[i:i+hist], day[i:i+hist], op[i:i+hist], promo[i:i+hist]]))
		Y.append(np.array([sales[i+hist], np.mean(sales[i:i+hist]), times[i+hist]]))

	X = np.vstack(X)
	Y = np.vstack(Y)
	cols = ['s%i' % i for i in range(hist)] + ['m%i' % i for i in range(hist)] + ['d%i' % i for i in range(hist)] + ['o%i' % i for i in range(hist)] + ['p%i' % i for i in range(hist)]
	cols.append('y')
	cols.append('mean')
	cols.append('time')
	df = pd.DataFrame(np.hstack([X,Y]), columns=cols)
	print(df)
	df.to_csv('trainfile.csv', index=False)

	max_time = np.max(df['time'])
	min_time = np.min(df['time'])
	ckpts = [min_time + j*(max_time-min_time)//splits for j in range(1, splits+1)]


	for i, ckpt in enumerate(ckpts):
		df_t = df[df['time'] <= ckpt]
		df_t.to_csv('train_%d.csv' % i, index=False)
		print(df_t.shape)
		df = df[df['time'] > ckpt]	

def make_split_sales(store_type, hist, splits):

	#stores = {'a': 2, 'b': 85, 'c': 1, 'd': 13}
	stores = {'a': 3, 'b': 259, 'c': 1, 'd': 13}
	store = stores[store_type]
	df = pd.read_csv('train.csv', parse_dates=True, index_col='Date')
	df = df.sort_index(ascending=True)
	df = df[(df.Store==store)]

	df['time'] = pd.Series(np.arange(df.shape[0], dtype=np.int32), index=df.index)
	df = df[df.Open != 0]
	df = df[['Sales', 'time']]
	print(df.index)
	print(df)

	times = df['time'].values
	sales = df['Sales'].values
	min_sales = np.min(sales)
	max_sales = np.max(sales)
	print('Max and min sales: (%d, %d)' % (max_sales,min_sales))
	sales = 100 * (sales - min_sales)/(max_sales - min_sales)
	X = list()
	Y = list()
	for i in range(sales.shape[0] - hist):

		X.append(sales[i:i+hist])
		Y.append(np.array([sales[i+hist], np.mean(sales[i:i+hist]), times[i+hist]]))

	X = np.vstack(X)
	Y = np.vstack(Y)
	cols = ['s%i' % i for i in range(hist)]
	cols.append('y')
	cols.append('mean')
	cols.append('time')
	df = pd.DataFrame(np.hstack([X,Y]), columns=cols)
	print(df)
	df.to_csv('trainfile.csv', index=False)

	max_time = np.max(df['time'])
	min_time = np.min(df['time'])
	ckpts = [min_time + j*(max_time-min_time)//splits for j in range(1, splits+1)]


	for i, ckpt in enumerate(ckpts):
		df_t = df[df['time'] <= ckpt]
		df_t.to_csv('train_%d.csv' % i, index=False)
		print(df_t.shape)
		df = df[df['time'] > ckpt]

	'''
	df = pd.read_csv('test.csv', parse_dates=True, index_col='Date')
	df = df.sort_index(ascending=True)
	df = df[(df.Store==store)]

	df['time'] = pd.Series(np.arange(df.shape[0], dtype=np.int32), index=df.index)
	df = df[df.Open != 0]
	df = df[['Sales', 'time']]
	print(df.index)
	print(df)

	times = df['time'].values
	sales = df['Sales'].values
	sales = 100 * (sales - min_sales)/(max_sales - min_sales)
	X = list()
	Y = list()
	for i in range(sales.shape[0] - hist):

		X.append(sales[i:i+hist])
		Y.append(np.array([sales[i+hist], np.mean(sales[i:i+hist]), times[i+hist]]))

	X = np.vstack(X)
	Y = np.vstack(Y)
	cols = ['s%i' % i for i in range(hist)]
	cols.append('y')
	cols.append('mean')
	cols.append('time')
	df = pd.DataFrame(np.hstack([X,Y]), columns=cols)
	print(df)
	df.to_csv('testfile.csv', index=False)
	'''

def main():


	parser = argparse.ArgumentParser()
	parser.add_argument('-t', required=True, metavar='type', help='Type')
	parser.add_argument('-n', metavar='csv_file', help='History size')
	parser.add_argument('-s', metavar='csv_file', help='No of splits')
	args = parser.parse_args()
	if args.n == None:
		args.n = 5
	if args.s == None:
		args.s = 5

	make_split(args.t, int(args.n), int(args.s))

if __name__ == '__main__':
	main()