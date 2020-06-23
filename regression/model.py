import tensorflow as tf
import numpy as np

def model(input_tensor, **kwargs):

	img_height = 100
	img_width = 100
	

	with tf.variable_scope('deep_cnn_layer', reuse=tf.AUTO_REUSE):

		
		feature = tf.compat.v1.layers.dense(input_tensor, 64, activation=tf.nn.relu, name='fc-1')
		feature = tf.compat.v1.layers.dense(feature, 32, activation=tf.nn.relu, name='fc-2')
		
		output_tensor = tf.compat.v1.layers.dense(feature, 1, name='fc-3')
		
	return output_tensor

def loss(y_true, y_pred):

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
	return loss

if __name__=="__main__":

	x = np.random.random_integers(0,9,900000).reshape((90, 100, 100, 1))
	y = np.random.random_integers(0,1,90).reshape((90,1))

	print(x.shape)
	print(y.shape)

	input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 100, 100, 1))
	label_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,1))

	train_ds = tf.data.Dataset.from_tensor_slices((input_tensor, label_tensor)).shuffle(100).batch(10).repeat()

	iterator = train_ds.make_initializable_iterator()
	img, lab = iterator.get_next()

	op = model(img)
	l = loss(lab, op)
	train_op = tf.train.AdamOptimizer().minimize(l)
	with tf.Session() as sess:

		sess.run(tf.compat.v1.global_variables_initializer())
		sess.run(iterator.initializer, feed_dict={input_tensor: x, label_tensor: y})
		for i in range(100):
			tot_loss = 0
			for _ in range(9):
				_, loss_value = sess.run([train_op, l])
				tot_loss += loss_value
			print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / 9))

	'''
	ip, op = model(x[0].shape)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		summary = sess.run(op, feed_dict={ip: x})
		print(summary.shape)
		print(summary)
	'''


	#for xx, yy in train_ds:
	#	print(xx.shape, yy.shape)
	#for i in train_ds.__iter__(): print(i)




def loss(y_true, y_pred):

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
	return loss





	