import numpy as np
import pygsvd
from scipy.linalg import null_space
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_moons
from sklearn.metrics import classification_report

class GFK():

	def __init__(self, dims=20, eps=1e-10):

		self.dims = dims
		self.eps = eps
		self.G_ = None
		self.classif_ = None
		self.X_s = None
		self.X_t = None

	def _get_principal_angle_matrices(self, P_s, P_t):

		Gamma = np.zeros((P_s.shape[1], P_t.shape[1]))
		Sigma = np.zeros((P_s.shape[1], P_t.shape[1]))

		

	def construct_geodesic_flow(self, P_s, P_t):

		dims = P_s.shape[1]
		R_s = null_space(P_s.T)
		N = dims + R_s.shape[1]

		#print(P_s.shape)
		#print(P_t.shape)
		#print(R_s.shape)

		A = np.matmul(P_s.T, P_t)
		B = np.matmul(R_s.T, P_t)

		
		Gam, Sig, V, U1, U2 = pygsvd.gsvd(A, B)
		U2 = -U2
		#print(Gam.shape)
		#print(Sig.shape)
		#print(V.shape)
		#print(U1.shape)
		#print(U2.shape)		

		theta = np.arccos(Gam)
		
		L1 = np.diag(0.5*(1 + (np.sin(2 * theta)/(2 * np.maximum(theta, self.eps)))))
		L2 = np.diag(0.5*((np.cos(2 * theta) - 1)/(2 * np.maximum(theta, self.eps))))
		L3 = L2
		L4 = np.diag(0.5*(1 - (np.sin(2 * theta)/(2 * np.maximum(theta, self.eps)))))

		print(L1.shape)
		print(L2.shape)
		print(L3.shape)
		'''
		omega1_1 = np.hstack([U1, np.zeros(shape=(dims, N-dims))])
		omega1_2 = np.hstack((np.zeros(shape=(N-dims, dims)), U2))
		omega1 = np.vstack([omega1_1, omega1_2])

		
		omega2_1 = np.hstack((L1, L2, np.zeros(shape=(dims, N-2*dims))))
		omega2_2 = np.hstack((L3, L4, np.zeros(shape=(dims, N-2*dims))))
		omega2_3 = np.zeros(shape=(N-2*dims, N))
		omega2 = np.vstack([omega2_1, omega2_2, omega2_3])
		'''
		omega1_1 = np.hstack([U1, np.zeros(shape=(dims, dims))])
		omega1_2 = np.hstack((np.zeros(shape=(N-dims, dims)), U2))
		omega1 = np.vstack([omega1_1, omega1_2])

		
		omega2_1 = np.hstack([L1, L2])
		omega2_2 = np.hstack([L2, L4])
		
		omega2 = np.vstack([omega2_1, omega2_2])

	
		omega3 = omega1.T

		omega = np.dot(np.dot(omega1, omega2), omega3)
		PR_s = np.hstack([P_s, R_s])
		self.G_ = np.dot(np.dot(PR_s, omega), PR_s.T)
		print(self.G_.shape)

		


	def GFK_kernel(self, X, Y):

		return np.dot(np.dot(X, self.G_), Y.T)



	def fit(self, X_source, X_target,  Y_source, Y_target=None):

		self.X_s = X_source
		self.X_t = X_target

		pca = PCA(n_components=self.dims)
		pca.fit(self.X_s)
		P_s = pca.components_.T
		pca.fit(self.X_t)
		P_t = pca.components_.T

		self.construct_geodesic_flow(P_s, P_t)

		#self.classif_ = SVC()
		self.classif_ = SVC(kernel=self.GFK_kernel)

		self.classif_.fit(X_source, Y_source)

	def predict(self, X_target, Y_target=None):

		Y_pred = self.classif_.predict(X_target)
		return Y_pred


if __name__== "__main__":

	
	### Random data
	X1, Y1 = make_classification(n_samples=500)
	X2, Y2 = make_classification(n_samples=500)

	gfk = GFK(dims=10)
	gfk.fit(X1, X2, Y1, Y2)
	Yp = gfk.predict(X2)

	print(classification_report(Y2, Yp))

	
	'''
	### 2-moons data
	X1, Y1 = make_moons(n_samples=500, noise=0.1)
	X2, Y2 = make_moons(n_samples=500, noise=0.1)
	X2=np.dot(X2, np.array([[0.86602540378, 0.5], [-0.5, 0.86602540378]]))


	gfk = GFK(dims=1)
	gfk.fit(X1, X2, Y1, Y2)
	Yp = gfk.predict(X2)

	print(classification_report(Y2, Yp))

	'''


		


