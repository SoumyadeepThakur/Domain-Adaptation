import numpy as np
from sklearn.decomposition import  PCA, IncrementalPCA
from gfk import GFK
from sklearn.svm import SVC, LinearSVR, SVR
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import classification_report

class CMA():

	def __init__(self, dims, block):

		self.dims = dims
		self.block = block
		self.X_s = None
		self.X_t = None
		self.classif_ = None
		self.Gs_ = None
		self.Xt_cgfk_ = None

	def cma_fit_predict(self, X_source, X_target, Y_source, Y_target=None):

		self.X_s = X_source
		self.X_t = X_target
		self.Gs_ = list()
		gfk = GFK(dims=self.dims)

		pca = PCA(n_components=self.dims)
		pca.fit(self.X_s)
		U = pca.components_.T

		self.Xt_cgfk_ = np.zeros(X_target.shape)
		
		incpca = IncrementalPCA(n_components=self.dims)

		for i in range(0,X_target.shape[0],self.block):

			print('Block',i)
			incpca.fit(self.X_t[i:i+self.block])
			P = incpca.components_.T

			gfk.construct_geodesic_flow(U, P)

			self.Gs_.append(gfk.G_)

			self.Xt_cgfk_[i:i+self.block] = np.dot(X_target[i:i+self.block], gfk.G_.T)
			#self.Xt_cgfk_[i:i+self.block] = X_target[i:i+self.block]

		#self.classif_ = SVC()
		#self.classif_.fit(X_source, Y_source)

		#Y_pred = self.classif_.predict(self.Xt_cgfk_)
		self.classif_ = LinearSVR(verbose=2)
		self.classif_.fit(X_source, Y_source)

		Y_pred = self.classif_.predict(self.Xt_cgfk_)

		return Y_pred

if __name__== "__main__":

	
	### Random data
	X1, Y1 = make_classification(n_samples=500)
	X2, Y2 = make_classification(n_samples=500)

	cma = CMA(dims=10, block=100)
	Yp = cma.cma_fit_predict(X1, X2, Y1, Y2)

	print(classification_report(Y2, Yp))

	
	'''
	### 2-moons data
	X1, Y1 = make_moons(n_samples=500, noise=0.1)
	X2, Y2 = make_moons(n_samples=500, noise=0.1)
	X2=np.dot(X2, np.array([[0.86602540378, 0.5], [-0.5, 0.86602540378]]))


	cma = CMA(dims=1, block=100)
	Yp = cma.cma_fit_predict(X1, X2, Y1, Y2)

	print(classification_report(Y2, Yp))

	'''



