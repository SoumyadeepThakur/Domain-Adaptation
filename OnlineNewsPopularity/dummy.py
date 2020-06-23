import numpy as np
import tensorflow as tf
from cvae import CVAE
from sklearn.datasets import make_classification, make_moons
import matplotlib.pyplot as plt

EPOCH = 50000
BATCH = 20


#X_tr, Y_tr = make_classification(n_samples=60000, n_features=100)
#Y_tr = Y_tr.reshape(-1,1)
#y = y.reshape(-1,1)

(X_tr, Y_tr), (X_te, Y_te) = tf.keras.datasets.mnist.load_data()
#plt.imshow(X_tr[2])
#plt.show()
#print(Y_tr[2])
X_tr = X_tr.reshape(-1,  784)/256.0
X_te =  X_te.reshape(-1,  784)/256.0
Y_tr = np.eye(10, dtype=np.int32)[Y_tr]
Y_te = np.eye(10, dtype=np.int32)[Y_tr]
#Y_tr = Y_tr.reshape(-1,1)
#Y_te = Y_te.reshape(-1,1)

print(X_tr.shape)
print(Y_tr.shape)
print(X_te.shape)
print(Y_te.shape)

input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 784))
cond_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 10))
label_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 784))
train_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=())

train_ds = tf.data.Dataset.from_tensor_slices((input_tensor, cond_tensor, label_tensor)).shuffle(60000).batch(BATCH).repeat()

iterator = train_ds.make_initializable_iterator()
dat, c, lab = iterator.get_next()

#cvae = CVAE(100, hidden_layers=1, dims=[50, 20])
cvae = CVAE(784, hidden_layers=1, dims=[196, 100])
#cvae.init_model(input_tensor, cond_tensor, label_tensor, train_tensor)
cvae.init_model(dat, c, lab, train_tensor)
model = cvae.model_

loss =  cvae.loss_

train_op = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:

	sess.run(tf.compat.v1.global_variables_initializer())
	sess.run(iterator.initializer, feed_dict={input_tensor: X_tr, cond_tensor: Y_tr, label_tensor: X_tr, train_tensor: 1})
	for i in range(EPOCH):
		tot_loss = 0
		for _ in range(X_tr.shape[1]//BATCH):
			_, loss_value = sess.run([train_op, loss])
								#feed_dict={input_tensor: X_tr, cond_tensor: Y_tr, label_tensor: X_tr, train_tensor: 1})
			tot_loss += loss_value
		if i%100 == 0:
			print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / (X_tr.shape[1]//BATCH)))

	yy = np.random.random_integers(0,9,10)
	print(yy)
	yy = np.eye(10)[yy]
	sess.run(iterator.initializer, feed_dict={input_tensor: X_te[0:10], cond_tensor: yy, label_tensor: X_te[0:10], train_tensor: 1})
	#for i in range(10):
		
	for img in sess.run(model['op_tensor']):
		print(img.shape)
		img = img.reshape(28,28)
		plt.imshow(img)
		plt.show()

		


