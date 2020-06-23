import numpy as np
import pandas as pd
import argparse

def split(filename, splits):

	name=filename.split('.')[0]
	df = pd.read_csv(filename)
	max_time = df['time'].max()
	print(max_time)
	ckpts = [j*max_time//splits for j in range(1, splits+1)]

	for i, ckpt in enumerate(ckpts):
		df_t = df[df['time'] <= ckpt]
		df_t.to_csv('%s_%d.csv' % (name, i), index=False)
		print(df_t.shape)
		df = df[df['time'] > ckpt]


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required=True, metavar='file', help='Input file name')
	parser.add_argument('-s', required=True, metavar='file', help='No of domains')
	args = parser.parse_args()
	split(args.f, int(args.s))
	

if __name__=="__main__":
	main()