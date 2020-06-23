import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def make_plot(filename, imagefile):

	lines = list()
	times = list()
	freq = list()

	with open(filename, 'r') as infile:
		lines = infile.readlines()
	
	for line in lines:
		times.append(int(line.rstrip().split()[1]))

	times = np.array(sorted(times))
	t = times[0]
	ind = 0

	while (t <= times[-1]):
		ind_next = np.searchsorted(times, t+3600)
		freq.append(times[ind:ind_next].shape[0])
		ind = ind_next
		t += 3600

	title = filename.split('.')[0]
	x = list(range(len(freq)))
	plt.bar(x,freq)
	plt.title(title)
	plt.xlabel('time')
	plt.ylabel('frequency')
	if imagefile == None:
		plt.show()
	else:
		plt.savefig(imagefile)

def make_plot_3(filename):

	df = pd.read_csv(filename)
	features = len(df.columns) - 3
	
	times = df['time'].values
	gaps = df['y'].values
	gaps_pred = df['y_pred'].values
	times_pred = times - gaps + gaps_pred
	
	print(np.mean(abs(times-times_pred)))
	t, t1 = times[0], times_pred[0]
	ind, ind1 = 0, 0
	freq = list()
	freq1 = list()
	while (t <= times[-1] and t1 <= times_pred[-1]):
		ind_next = np.searchsorted(times, t+3600)
		ind_next1 = np.searchsorted(times_pred, t1+3600)
		freq.append(ind_next - ind)
		freq1.append(ind_next1 - ind1)
		ind = ind_next
		ind1 =  ind_next1
		t += 3600
		t1 += 3600


	freq = np.array(freq)
	freq1 = np.array(freq1)
	x = np.arange(freq.shape[0])

	plt.plot(x, freq, label='true')
	plt.plot(x, freq1, label='predicted')
	plt.legend(loc="upper right")
	plt.xlabel('time in days')
	plt.ylabel('frequency of tweets')
	plt.title(filename.split('_')[0])
	plt.show()

def make_plot_2(filename):

	df = pd.read_csv(filename)
	features = len(df.columns) - 3
	time = df['time'].values
	init_gaps = df.iloc[0].values[0:features]
	init_times = np.zeros(shape=(features+1,))
	init_times[-1] = time[0] - df.iloc[0].values[-3]
	for i in range(features):
		init_times[-i-2] = init_times[-i-1] - init_gaps[-i-1]
	time = np.hstack([init_times, time])
	
	t=time[0]
	ind = 0
	freq = list()
	while (t <= time[-1]):
		ind_next = np.searchsorted(time, t+3600)
		freq.append(ind_next - ind)
		ind = ind_next
		t += 3600


	x = list(range(len(freq)))
	plt.bar(x,freq)
	
	plt.xlabel('time in days')
	plt.ylabel('frequency of tweets')

	plt.show()

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required=True, metavar='text_file', help='Input text file name')
	parser.add_argument('-o', metavar='image_file', help='Outfile image file name')
	args = parser.parse_args()
	#make_plot_2(args.f)
	make_plot_3(args.f)



if __name__ == "__main__":
	main()