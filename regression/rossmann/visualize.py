import argparse
import numpy as np
import pandas as pd
from pandas import datetime
import matplotlib
import matplotlib.pyplot as plt

def make_plot(filename, store_type):

	df = pd.read_csv(filename)
	stores = {'a': 3, 'b': 259, 'c': 1, 'd': 13}
	store = stores[store_type]


	df = pd.read_csv(filename, parse_dates=True, index_col='Date')
	df = df.sort_index(ascending=True)
	df = df[(df.Store==store)]

	df['time'] = pd.Series(np.arange(df.shape[0], dtype=np.int32), index=df.index)
	
	df = df[['Sales', 'time']]
	
	sales = df['Sales'].values
	times = df['time'].values

	df['Sales'].resample('W').sum().plot()
	plt.bar(times,sales)
	
	plt.xlabel('days')
	plt.ylabel('sales')

	plt.show()

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required=True, metavar='text_file', help='Input text file name')
	parser.add_argument('-t', metavar='store type', help='Store type')
	args = parser.parse_args()
	#make_plot_2(args.f)
	make_plot(args.f, args.t)



if __name__ == "__main__":
	main()