import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from transport import *
from model import *
from regression_grassmannian import * 
from cma import CMA
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def load_data(files_list):

	# Assuming the last to be target, and first all to be source
	
	for file in files_list:

		df = pd.read_csv(file)
		Y_temp = df['y'].values
		X_temp = df.drop(['y'], axis=1).values
		yield X_temp, Y_temp



def __dummy_train(X_train, Y_train, X_target, Y_target):

	svc = SVC(verbose=2)
	#svc=KNeighborsClassifier(n_neighbors=1)
	#svc = MLPClassifier((64, 32), verbose=2)
	#svc = LinearSVR(verbose=2, max_iter=1000)
	#svc = MLPRegressor((32,16), learning_rate_init=0.1, verbose=2)
	#rfc = RandomForestClassifier(n_estimators=20, verbose=2)
	svc.fit(X_train, Y_train)
	Y_pred = svc.predict(X_target)

	print(accuracy_score(Y_target, Y_pred))
	


def main():

	# Include as args
	#files = ['news_0.csv', 'news_1.csv', 'news_2.csv', 'news_3.csv','news_4.csv']
	train_files = ['moon_%d.csv' % i for i in range(11)]
	test_files = ['moon_test_%d.csv' % i for i in range(1,11)]
	
	X_train, X_test = [], []
	Y_train, Y_test = [], []
	count=0
	
	for X, Y in load_data(train_files):
		print(X.shape)
		print(Y.shape)

		X_train.append(X)
		Y_train.append(Y)

	for X, Y in load_data(test_files):
		print(X.shape)
		print(Y.shape)

		X_test.append(X)
		Y_test.append(Y)

	
	X_trans, X_trans_list, X_trans_1 = transform_samples(X_train[0], Y_train[0], X_train[1:-1], Y_train[1:-1],
										 X_train[-1], Y_train[-1], time_reg=True)
	
	X_train = [X_trans]
	X_train = X_train + X_trans_list + [X_trans_1]
	
	#print(len(X_train))
	for i in range(1,11):
	
		__dummy_train(X_train[i], Y_train[0], X_test[i-1], Y_test[i-1])
	
	#train(X_train, Y_train)
	#test(X_target, Y_target)
	
	
	
if __name__ == "__main__":
	main()