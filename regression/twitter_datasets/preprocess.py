import numpy as np
import pandas as pd
import argparse

def make_csv_time(filename, csvfile, length, splits):

	lines = list()
	times = list()

	with open(filename, 'r') as infile:
		lines = infile.readlines()
	
	for line in lines:
		split_line = line.rstrip().split()
		times.append(int(split_line[1]))

	times = np.sort(np.array(times))
	times = times - times[0]
	gaps = np.diff(times)

	print('No of gaps: ', gaps.shape[0])
	print('Min gap: ', np.min(gaps))
	print('Max gap: ',np.max(gaps))
	print('Mean gap: ',np.mean(gaps))
	print('Stddev gap: ',np.std(gaps))

	X = list()
	Y = list()
	for i in range(gaps.shape[0]-length):

		X.append(gaps[i: i+length])
		Y.append(np.hstack([gaps[i+length], times[i+length+1], np.mean(gaps[i:i+length])]))

	X = np.vstack(X)
	Y = np.vstack(Y)

	print(X.shape)
	print(Y.shape)
	cols = ['g%s' % i for i in range(1,length+1)]
	cols.append('y')
	cols.append('time')
	cols.append('mean')

	print('Splitting into domains')
	df = pd.DataFrame(data=np.hstack([X, Y]), columns=cols)
	print(df)
	print('Writing to ', csvfile)
	df.to_csv(csvfile, index=False)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required=True, metavar='text_file', help='Input text file name')
	parser.add_argument('-o', metavar='csv_file', help='Outfile csv file name')
	parser.add_argument('-n', metavar='csv_file', help='No of features')
	parser.add_argument('-s', metavar='csv_file', help='No of splits')
	args = parser.parse_args()
	if args.n == None:
		args.n = 5
	if args.s == None:
		args.s = 5

	make_csv_time(args.f, args.o, int(args.n), int(args.s))

if __name__=="__main__":
	main()