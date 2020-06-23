import numpy as np
import pandas as pd
import tensorflow as tf
from ot.da import *
from cvae import CVAE

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from transport import *
from model import *

EPOCH = 20
BATCH = 32

def load_data(files_list):

	# Assuming the last to be target, and first all to be source
	
	for file in files_list:

		df = pd.read_csv(file)
		Y_temp = df[' shares'].values
		X_temp = df.drop([' timedelta', ' shares'], axis=1).values
		Y_temp = np.array([0 if d<=1400 else 1 for d in Y_temp])
		yield X_temp, Y_temp

def preprocess(X_source, X_aux_list, X_target):

	scaler = MinMaxScaler()
	X_source = scaler.fit_transform(X_source)

	for i in range(len(X_aux_list)):
		X_aux_list[i] = scaler.transform(X_aux_list[i])

	X_target = scaler.transform(X_target)

	return X_source, X_aux_list, X_target

def train_cvae(X_source, X_aux_list, X_target, X_source_t, X_aux_list_t, X_target_t):

	cvae = CVAE(input_shape=58, hidden_layers=1, dims=[45, 30])
	
	input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 58))
	cond_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 4))
	label_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 58))
	train_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=())	

	train_ds = tf.data.Dataset.from_tensor_slices((input_tensor, cond_tensor, label_tensor)).shuffle(100000).batch(BATCH).repeat()

	iterator = train_ds.make_initializable_iterator()
	dat, c, lab = iterator.get_next()

	#cvae = CVAE(100, hidden_layers=1, dims=[50, 20])
	#cvae.init_model(input_tensor, cond_tensor, label_tensor, train_tensor)


	cvae.init_model(dat, c, lab, train_tensor)
	model = cvae.model_

	loss =  cvae.loss_

	train_op = tf.train.AdamOptimizer().minimize(loss)



	X_train = np.vstack([X_source] + X_aux_list)
	X_train_t = np.vstack([X_source_t] + X_aux_list_t)
	k = len(X_aux_list)+1
	C_train = [k]*X_source.shape[0]
	for i in range(len(X_aux_list)):
		k-=1
		C_train += [k]*X_aux_list[i].shape[0]

	print(len(C_train))
	print(X_train.shape)
	print(C_train)

	C_train = np.array(C_train)
	C_train = np.eye(len(X_aux_list)+1)[C_train-1]

	with tf.Session() as sess:

		sess.run(tf.compat.v1.global_variables_initializer())
		sess.run(iterator.initializer, feed_dict={input_tensor: X_train, cond_tensor: C_train, label_tensor: X_train_t, train_tensor: 1})
		for i in range(EPOCH):
			tot_loss = 0
			for _ in range(X_train.shape[1]//BATCH):
				_, loss_value = sess.run([train_op, loss])
									#feed_dict={input_tensor: X_train, cond_tensor: Y_tr, label_tensor: X_train, train_tensor: 1})
				tot_loss += loss_value
			#if i%100 == 0:
			print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / (X_train.shape[1]//BATCH)))
		
		yy = np.random.random_integers(0,3,X_target.shape[0])
		print(yy)
		yy = np.eye(len(X_aux_list)+1)[yy]
		l=[]
		sess.run(iterator.initializer, feed_dict={input_tensor: X_target, cond_tensor: yy, label_tensor: X_target_t, train_tensor: 1})
		for i in range(2000):
			
			for img in sess.run(model['op_tensor']):
				l.append(np.array(img))

		l = np.vstack(l)
		np.savetxt('abc.txt', l)

def main():

	# Include as args
	files = ['news_0.csv', 'news_1.csv', 'news_2.csv', 'news_3.csv','news_4.csv']
	
	X_source, X_aux_list, X_target = [], [], []
	Y_source, Y_aux_list, Y_target = [], [], []
	count=0
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

	#X_aux = np.vstack(X_aux)
	#Y_aux = np.hstack(Y_aux)

	#X_source, X_aux_list, X_target = preprocess(X_source, X_aux_list, X_target)


	#print(X_aux.shape)
	#print(Y_aux.shape)

	## Transport part

	X_source_t, X_aux_list_t, X_target_t = transform_samples_otda(X_source, Y_source, X_aux_list, Y_aux_list, X_target, Y_target)
	print(X_source.shape)
	print(X_source_t.shape)
	print(X_aux_list[0].shape)
	print(X_aux_list_t[0].shape)
	print(X_aux_list[1].shape)
	print(X_aux_list_t[1].shape)
	print(X_aux_list[2].shape)
	print(X_aux_list_t[2].shape)
	print(X_target.shape)
	print(X_target_t.shape)
	X_source, X_aux_list, X_target = preprocess(X_source, X_aux_list, X_target)
	X_source_t, X_aux_list_t, X_target_t = preprocess(X_source_t, X_aux_list_t, X_target_t)

	X_train = np.vstack([X_source] + X_aux_list)
	X_train_t = np.vstack([X_source_t] + X_aux_list_t)
	Y_train = np.hstack([Y_source] + Y_aux_list)

	# CVAE part

		

	train_cvae(X_source, X_aux_list, X_target, X_source_t, X_aux_list_t, X_target_t)

	#train(X_train, Y_train)
	#test(X_target, Y_target)

if __name__=="__main__":

	main()

