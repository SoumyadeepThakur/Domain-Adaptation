import numpy as np
import pandas as pd
import argparse

def make_csv(filename, csvfile):

	lines = list()
	times = list()
	f1 = list()
	f2 = list()
	freq = list()

	with open(filename, 'r') as infile:
		lines = infile.readlines()
	
	for line in lines:
		split_line = line.rstrip().split()
		times.append(int(split_line[1]))
		f1.append(int(split_line[0]))
		f2.append(float(split_line[2]))

	times = np.array(times)
	f1 = np.array(f1)
	f2 = np.array(f2)
	indices = np.argsort(times)
	times = times[indices]
	f1 = f1[indices]
	f2 = f2[indices]

	#Scale 
	f1 = (f1-f1.mean())/f1.std()
	f2 = (f2-f2.mean())/f2.std()
	times  = times - times[0]
	print(np.min(times))
	print(np.max(times))
	print(times[0], times[-1])
	print(times.shape[0])
	print(f1.shape[0])
	F = list()
	Y = list()
	for i in range(times.shape[0]-5):

		F.append(np.hstack([f1[i:i+5], f2[i:i+5], times[i:i+5]]))
		Y.append(times[i+5])

	F = np.vstack(F)
	Y = np.hstack(Y)

	print(F.shape)
	print(Y.shape)

	print(F[0], Y[0])
	print(F[1], Y[1])
	print(F[2], Y[2])
	cols = ['f11', 'f12','f13', 'f14', 'f15', 'f21', 'f22', 'f23', 'f24', 'f25', 't1' ,'t2', 't3' ,'t4' ,'t5', 'next']
	df = pd.DataFrame(data=np.hstack([F, Y.reshape(-1,1)]), columns=cols)
	print(df)
	df.to_csv(csvfile, index=False)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required=True, metavar='text_file', help='Input text file name')
	parser.add_argument('-o', metavar='csv_file', help='Outfile csv file name')
	args = parser.parse_args()
	make_csv(args.f, args.o)

if __name__=="__main__":
	main()

