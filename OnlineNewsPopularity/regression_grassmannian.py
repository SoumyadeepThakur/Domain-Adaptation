import numpy as np
from gfk import *
from sklearn.decomposition import PCA
from sklearn.svm import *
class RegressionGrassmannian():

	def __init__(self, dims, eta):

		self.dims = dims
		self.P_s = None
		self.P_t = None
		self.P_a = list()
		self.eta = eta

	def rbf(self, s, lim):

		x = np.arange(lim+1)
		K_s = np.exp((lim-x)**2)
		K = np.exp((s-lim)**2)/np.sum(K_s)
		return K


	def regress(self, X_source, Y_source, X_aux_list, X_target):

		self.X_source = X_source
		self.Y_source = Y_source
		self.X_aux_list = X_aux_list
		self.X_target = X_target

		pca = PCA(n_components=self.dims)
		
		pca.fit(self.X_source)
		self.P_s = pca.components_.T
		for i in range(len(self.X_aux_list)):
			pca.fit(self.X_aux_list[i])
			self.P_a.append(pca.components_.T)

		pca.fit(self.X_target)
		self.P_t = pca.components_.T

		

		for k in range(1):

			grad = -2*self.rbf(0, len(X_aux_list)+1)*np.matmul(self.P_s,np.linalg.inv(np.matmul(self.P_t.T, self.P_s)))
			for i in range(len(self.X_aux_list)):
				grad -= 2*self.rbf(i+1, len(X_aux_list)+1)*np.matmul(self.P_a[i],np.linalg.inv(np.matmul(self.P_t.T, self.P_a[i])))

			A = np.matmul(grad, self.P_t.T) - np.matmul(self.P_t, grad.T)
			
			I = np.eye(A.shape[0])
			self.P_t = np.matmul(np.matmul(np.linalg.inv(I + 0.5*self.eta*A), I-0.5*self.eta*A), self.P_t)

		


	def fit_predict(self, Y_target):

		gfk = GFK(self.dims)
		gfk.construct_geodesic_flow(self.P_s, self.P_t)

		X_t_gfk = np.matmul(self.X_target, gfk.G_.T)

		#self.classif_ = SVC()
		self.classif_ = LinearSVR(verbose=1)
		self.classif_.fit(self.X_source, self.Y_source)

		Y_pred = self.classif_.predict(X_t_gfk)
		#print(classification_report(Y_target, Y_pred))
		
		return Y_pred




		
			




		