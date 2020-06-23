import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from transport import *
from model import *
from regression_grassmannian import * 
from cma import CMA
from sklearn.neural_network import MLPRegressor

def load_data(files_list):

	# Assuming the last to be target, and first all to be source
	
	for file in files_list:

		df = pd.read_csv(file)
		Y_temp = df[' shares'].values
		X_temp = df.drop([' timedelta', ' shares'], axis=1).values
		Y_temp = np.array([0 if d<=1400 else 1 for d in Y_temp])
		print(Y_temp[Y_temp==0].shape)
		print(Y_temp[Y_temp==1].shape)
		yield X_temp, Y_temp

def load_data_2(files_list):

	for file in files_list:

		df = pd.read_csv(file)
		Y_temp = df['next'].values
		X_temp = df.drop(['next'], axis=1).values
		
		yield X_temp, Y_temp

def preprocess(X_source, X_aux_list, X_target):

	scaler = MinMaxScaler()
	X_source = scaler.fit_transform(X_source)

	for i in range(len(X_aux_list)):
		X_aux_list[i] = scaler.transform(X_aux_list[i])

	X_target = scaler.transform(X_target)

	return X_source, X_aux_list, X_target

def __dummy_train(X_train, Y_train, X_target, Y_target):

	svc = SVC(kernel='linear', verbose=2)
	#svc = MLPClassifier((64, 32), verbose=2)
	#svc = LinearSVR(verbose=2, max_iter=1000)
	#svc = MLPRegressor((32,16), learning_rate_init=0.1, verbose=2)
	#rfc = RandomForestClassifier(n_estimators=20, verbose=2)
	svc.fit(X_train, Y_train)
	Y_pred = svc.predict(X_target)

	print(classification_report(Y_target, Y_pred))

def train(X_train, Y_train):

	
	perm = np.array(np.arange(X_train.shape[0]))
	np.random.shuffle(perm)

	X_train = X_train[perm]
	Y_train = Y_train[perm]

	Y_train = Y_train[:, np.newaxis]


	print(X_train.shape)
	print(Y_train.shape)

	input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, X_train.shape[1]))
	label_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,1))

	train_ds = tf.data.Dataset.from_tensor_slices((input_tensor, label_tensor)).shuffle(10000).batch(100).repeat()

	iterator = train_ds.make_initializable_iterator()
	in_vector, label = iterator.get_next()

	op_vector = model(in_vector)
	l = loss(label, op_vector)
	train_op = tf.train.AdamOptimizer().minimize(l)

	with tf.Session() as sess:

		sess.run(tf.compat.v1.global_variables_initializer())
		sess.run(iterator.initializer, feed_dict={input_tensor: X_train, label_tensor: Y_train})
		for i in range(1000):
			tot_loss = 0
			for _ in range(X_train.shape[0]//100):
				_, loss_value = sess.run([train_op, l])
				tot_loss += loss_value
			print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / (X_train.shape[0]//100)))

def test(X_test, Y_test):

	input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, X_test.shape[1]))
	label_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,1))
	Y_test = Y_test[:, np.newaxis]
	op_vector = model(input_tensor)
	with tf.Session() as sess:

		sess.run(tf.compat.v1.global_variables_initializer())
		Y_pred = sess.run(op_vector, feed_dict={input_tensor: X_test, label_tensor: Y_test})

	Y_pred = np.where(Y_pred>=0.5, 1, 0)
	print(Y_pred.shape)
	print(classification_report(Y_test, Y_pred))
	

def train_mrg(X_source, Y_source, X_aux_list, Y_aux_list, X_target, Y_target):

	rg = RegressionGrassmannian(7, 0.01)
	rg.regress(X_source, Y_source, X_aux_list, X_target)
	Y_pred = rg.fit_predict(Y_target)
	print(((Y_pred - Y_target)**2).mean())
	print(((Y_pred - Y_target)**2).max())
	print(((Y_pred - Y_target)**2).min())
	print(((Y_pred - Y_target)**2).std())

def train_cma(X_source, Y_source, X_aux_list, Y_aux_list, X_target, Y_target):

	X_train = np.vstack([X_source] + X_aux_list)
	Y_train = np.hstack([Y_source] + Y_aux_list)
	print('shapes')
	print(X_train.shape)
	print(Y_train.shape)
	print(X_target.shape)
	print(Y_target.shape)

	cma = CMA(dims=7, block=200)
	Y_pred = cma.cma_fit_predict(X_train, X_target, Y_train, Y_target)

	#print(classification_report(Y_target, Yp))

	print((abs(Y_pred - Y_target)).mean())
	print((abs(Y_pred - Y_target)).max())
	print((abs(Y_pred - Y_target)).min())
	print((abs(Y_pred - Y_target)).std())


def main():

	# Include as args
	files = ['news_0.csv', 'news_1.csv', 'news_2.csv', 'news_3.csv','news_4.csv']
	
	X_source, X_aux_list, X_target = [], [], []
	Y_source, Y_aux_list, Y_target = [], [], []
	count=0
	#for X, Y in load_data(files):
	for X, Y in load_data(files):
		print(X.shape)
		print(Y.shape)

		if count == 0:
			X_source = X
			Y_source = Y

		elif count == len(files) - 1:
			X_target = X
			Y_target = Y

		else:
			X_aux_list.append(X)
			Y_aux_list.append(Y)

		count+=1

	#train_cma(X_source, Y_source, X_aux_list, Y_aux_list, X_target, Y_target)

	
	X_source, X_aux_list, X_target = transform_samples_iter_reg(X_source, Y_source, X_aux_list, Y_aux_list, X_target, Y_target)
	
	print(X_source.shape)
	print(Y_source.shape)
	print(X_target.shape)
	print(Y_target.shape)
	
	#X_source, X_aux_list, X_target = preprocess(X_source, X_aux_list, X_target)
	
	X_train = np.vstack([X_source] + X_aux_list)
	Y_train = np.hstack([Y_source] + Y_aux_list)
	
	__dummy_train(X_train, Y_train, X_target, Y_target)
	
	#train(X_train, Y_train)
	#test(X_target, Y_target)
	
	
	
if __name__ == "__main__":
	main()