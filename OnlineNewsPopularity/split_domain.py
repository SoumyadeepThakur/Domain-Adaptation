import numpy as np
import pandas as pd
import argparse

def split(filename):

	df = pd.read_csv(filename)
	df = df.drop(['url'], axis=1)
	max_time = df[' timedelta'].max()
	ckpts = [150, 300, 450, 600, max_time]

	for i, ckpt in enumerate(ckpts):
		df_t = df[df[' timedelta'] <= ckpt]
		df_t.to_csv('news_%d.csv' % i, index=False)
		print(df_t.shape)
		df = df[df[' timedelta'] > ckpt]


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required=True, metavar='file', help='Input file name')
	args = parser.parse_args()
	split(args.f)
	

if __name__=="__main__":
	main()