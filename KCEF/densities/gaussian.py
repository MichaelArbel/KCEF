import numpy as np
from KCEF.densities.base_density import BaseDensity


class Gaussian(BaseDensity):
	def __init__(self, sigma = 1. , D=1):
		BaseDensity.__init__(self, D)
		self.sigma = sigma
		self.name  = "Gaussian" 
	def log_pdf(self, X):
		raise NotImplementedError()
	def grad(self, X):
		raise NotImplementedError()
	def sample(self, N=1):
		raise NotImplementedError()

	def quad_part(self):
		raise NotImplementedError()



class IsotropicCenteredGaussian(Gaussian):
	def __init__(self, sigma=2., D=1):
		Gaussian.__init__(self, sigma, D)
		self.name = "IsotropicCenteredGaussian"

	def log_pdf(self, X):
		if len(X.shape) < 2:
			X = np.reshape(X, [1,-1])
		quadratic_part = - 0.5/self.sigma**2 * np.sum(np.multiply(X,X), axis=1)
		const_part     = -self.D * np.log(self.sigma) - 0.5*self.D * np.log(2*np.pi)

		return const_part  + quadratic_part

	def grad(self, X):
		return - X / (self.sigma**2)

	def sample(self, N = 1):
		return np.random.randn(N,self.D)*self.sigma
	def quad_part(self, dim, num):
		quad = np.zeros([dim, dim, num])
		for i in range(num):
			quad[:,:,i] = (1./self.sigma)*np.eye(dim)

		return quad
