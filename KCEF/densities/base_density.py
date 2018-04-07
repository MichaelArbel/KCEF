import numpy as np

class BaseDensity(object):
	def __init__(self, D=1):
		self.D= D 
		self.name = "BaseDensity"
	def log_pdf(self, X):
		raise NotImplementedError()
	def grad(self, X):
		raise NotImplementedError()
	def sample(self, N=1):
		raise NotImplementedError()
	def quad_part(self):
		raise NotImplementedError()

class FlatDensity(BaseDensity):
	def __init__(self, D=1):
		BaseDensity.__init__(self, D)
		self.name = "FlatDensity"
	def log_pdf(self, X):
		return 0.
	def grad(self, X):
		return np.zeros(X.shape)
	def sample(self, N=1):
		return None
	def quad_part(self, dim, num):
		quad = np.zeros([dim, dim, num])
		return quad