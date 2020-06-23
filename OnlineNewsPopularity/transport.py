import numpy as np
import ot
from regularized_ot import RegularizedSinkhornTransport, RegularizedSinkhornTransportOTDA

def transform_samples(X_source, Y_source, X_aug_list, Y_aug_list, X_target, Y_target, time_reg=False):

	ot_sinkhorn = RegularizedSinkhornTransport(reg_e=0.5, alpha=10, max_iter=100, norm="median", verbose=True)

	# Source

	X_domain = [X_source]
	X_domain = X_domain + X_aug_list + [X_target]
	Y_domain = [Y_source]
	Y_domain = Y_domain + Y_aug_list + [Y_target]

	print(len(X_domain))

	i=0

	print('Domain %d' % i)
	print('Shape', X_domain[i].shape)
	gamma = []
	for j in range(i+1,len(X_domain)):

		print('Transforming to %d' % j)
		if time_reg:
			if j==i+1: ot_sinkhorn.fit(Xs=X_domain[i], ys=Y_domain[i], Xt=X_domain[j], yt=Y_domain[j], iteration=0)
			else: ot_sinkhorn.fit(Xs=X_domain[i], ys=Y_domain[i], Xt=X_domain[j], yt=Y_domain[j], prev_gamma=gamma, iteration=1)
		else:	
			ot_sinkhorn.fit(Xs=X_domain[i], ys=Y_domain[i], Xt=X_domain[j], yt=Y_domain[j], iteration=0)
		gamma = ot_sinkhorn.coupling_
		X_domain[j] = ot_sinkhorn.transform(X_domain[i])

	return X_domain[0], X_domain[1:-1], X_domain[-1]



def transform_samples_reg(X_source, Y_source, X_aug_list, Y_aug_list, X_target, Y_target, time_reg=True):

	ot_sinkhorn_r = RegularizedSinkhornTransport(reg_e=0.5, alpha=10, max_iter=100, norm="median", verbose=True)

	X_domain = [X_source]
	X_domain = X_domain + X_aug_list + [X_target]
	Y_domain = [Y_source]
	Y_domain = Y_domain + Y_aug_list + [Y_target]

	'''
	ot_sinkhorn_r.fit(Xs=X_domain[0], ys=Y_domain[0], Xt=X_domain[1], yt=Y_domain[1], iteration=0)
	tx = ot_sinkhorn_r.coupling_
	print('Gamma shape: ',tx.shape)
	ot_sinkhorn_r.fit(Xs=X_domain[1], ys=Y_domain[1], Xt=X_domain[2], yt=Y_domain[2], prev_gamma=tx, iteration=1)
	'''
	i=0
	

	print('Domain %d' % i)
	print('Shape', X_domain[i].shape)

	gamma = []
	X_temp = X_domain[i]
	for j in range(i+1,len(X_domain)):

		print('Transforming to %d' % j)

		if time_reg:
			if j==i+1: ot_sinkhorn_r.fit(Xs=X_domain[i], Xt=X_domain[j], ys=Y_domain[i], yt = Y_domain[j], iteration=0)
			else: ot_sinkhorn_r.fit(Xs=X_domain[j-1], Xt=X_domain[j], ys=Y_domain[i], yt=Y_domain[j], prev_gamma=gamma, iteration=1)
		else:
			ot_sinkhorn_r.fit(Xs=X_domain[j-1], Xt=X_domain[j], ys=Y_domain[i], yt = Y_domain[j], iteration=0)	
		gamma = ot_sinkhorn_r.coupling_
		X_domain[j] = ot_sinkhorn_r.transform(X_domain[j-1])

	#X_domain[i] = X_temp
	return X_domain[0], X_domain[1:-1], X_domain[-1]	


def transform_samples_iter_reg(X_source, Y_source, X_aug_list, Y_aug_list,  X_target, Y_target, time_reg=True):

	ot_sinkhorn_r = RegularizedSinkhornTransport(reg_e=0.5, alpha=10, max_iter=100, norm="median", verbose=True)

	X_domain = [X_source]
	X_domain = X_domain + X_aug_list + [X_target]
	Y_domain = [Y_source]
	Y_domain = Y_domain + Y_aug_list + [Y_target]

	'''
	ot_sinkhorn_r.fit(Xs=X_domain[0], ys=Y_domain[0], Xt=X_domain[1], yt=Y_domain[1], iteration=0)
	tx = ot_sinkhorn_r.coupling_
	print('Gamma shape: ',tx.shape)
	ot_sinkhorn_r.fit(Xs=X_domain[1], ys=Y_domain[1], Xt=X_domain[2], yt=Y_domain[2], prev_gamma=tx, iteration=1)
	'''
	i=0
	

	print('Domain %d' % i)
	print('Shape', X_domain[i].shape)

	gamma = []
	
	for i in range(len(X_domain)):

		X_temp = X_domain[i]

		for j in range(i+1,len(X_domain)):

			print('Transforming to %d' % j)

			if time_reg:
				if j==i+1: ot_sinkhorn_r.fit(Xs=X_domain[i], Xt=X_domain[j], ys=Y_domain[i], yt = Y_domain[j], iteration=0)
				else: ot_sinkhorn_r.fit(Xs=X_temp, Xt=X_domain[j], ys=Y_domain[i], yt=Y_domain[j], prev_gamma=gamma, iteration=1)
			else:
				ot_sinkhorn_r.fit(Xs=X_temp, Xt=X_domain[j], ys=Y_domain[i], yt = Y_domain[j], iteration=0)	
			gamma = ot_sinkhorn_r.coupling_
			X_temp = ot_sinkhorn_r.transform(X_temp)
		
		X_domain[i] = X_temp

	#X_domain[i] = X_temp
	return X_domain[0], X_domain[1:-1], X_domain[-1]	
	
def transform_samples_reg_otda(X_source, Y_source, X_aug_list, Y_aug_list,  X_target, Y_target):

	ot_sinkhorn_r = RegularizedSinkhornTransportOTDA(reg_e=0.5, max_iter=50, norm="median", verbose=True)

	X_domain = [X_source]
	X_domain = X_domain + X_aug_list + [X_target]
	Y_domain = [Y_source]
	Y_domain = Y_domain + Y_aug_list + [Y_target]

	for i in range(len(X_domain) - 1):

		print('Domain %d' % i)
		print('Shape', X_domain[i].shape)

		gamma = []
		X_temp = X_domain[i]
		Y_temp = Y_domain[i]

		for j in range(i+1, len(X_domain)):

			print('Transforming to %d' % j)

			if j==i+1: ot_sinkhorn_r.fit(Xs=X_domain[j-1], ys=Y_domain[j-1], Xt=X_domain[j], yt=Y_domain[j], 
											Xs_trans=X_temp, ys_trans=Y_domain[i], iteration=0)
			else: ot_sinkhorn_r.fit(Xs=X_domain[j-1], ys=Y_domain[j-1], Xt=X_domain[j], yt=Y_domain[j],
											Xs_trans=X_temp, ys_trans=Y_domain[i], prev_gamma=gamma, iteration=1)
			gamma = ot_sinkhorn_r.coupling_
			X_temp = ot_sinkhorn_r.transform(X_temp)

		X_domain[i] = X_temp

	return X_domain[0], X_domain[1:-1], X_domain[-1]
	return X_domain[0], X_domain[1:-1], X_domain[-1]