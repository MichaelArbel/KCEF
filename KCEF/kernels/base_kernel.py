import numpy as np

class BaseKernel(object):
	def __init__(self, D):
		self.D = D 
		self.params =1.
		self.isNull = False
 	
 	def set_params(self,params):
		raise NotImplementedError()
	def get_params(self):
		raise NotImplementedError()

	def kernel(self, X,basis=None):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix
		raise NotImplementedError()

	def weighted_kernel(self,Y, basis , K):
		# Computes the Hadamard product between the gramm matrix of (Y, basis) and the matrix K
		# Input:
		# Y 	: N by d matrix of data points
		# basis : M by d matrix of basis points		
		# K 	: N by M matrix
		# output: N by M matrix   K had_prod Gramm(Y, basis)
		raise NotImplementedError()
	def hessian_bloc(self,a,b,Y_a, Y_b,k,d):

		# Partial weighted Hessian of k(Y_1, Y_2) output = K_X(X_i, X_j) kro_prod  \partial_i \partial_i k(Y_i,Y_j) for fixed i and j
		# Inputs
		# Y_i, Y_j  : N by 1  slices of data points on dimensions i and j , N is the number of data points
		# K_X 		: N by N matrix
		# output    : N by N matrix 
		raise NotImplementedError()



	def hessian(self,Y, K_X):
		# weighted hessian of   k(Y_1, Y_2)  : output = K_X(X_i, X_j) kro_prod  \partial_i \partial_i k(Y_i,Y_j) for  1 < i, j <d 
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# K_X 		: N by N matrix
		# output    : Nd by Nd matrix 
		raise NotImplementedError()

	def w_grad(self, Y, basis,  beta , K):

		# computes grad weighted_kernel(Y, basis, K_X) with respect to basis, multiplies by beta and sums over basis:  sum_b grad weighted_kernel(Y, basis_b, K_X) beta_b
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# beta 		: M by d matrix of weights
		# K_X 		: N by M matrix
		# output    : N by 1 vector   

		raise NotImplementedError()
	
	def laplacian(self, Y, basis,  K ):
		# Computes the Laplacian of weighted_kernel(Y, basis, K) with respect to basis and sum over basis:    \sum_{basis}   Delta_{basis} weighted_kernel(Y, basis, K)      
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# K_X 		: N by M matrix
		# output    : N by 1 vector   	
		raise NotImplementedError()
	
	def grad_laplacian(self,Y, basis, K ):
		# Computes the jacobian with respect to Y of laplacian(Y, basis, K_X) :   grad_Y laplacian(Y, basis, K_X)
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# K_X 		: N by M matrix
		# output    : N by d vector  
		raise NotImplementedError()
	
	def w_cross_hessian(self, Y, basis, beta,  K):
		# Computes the product of beta_n with  \partial_Y\partial_beta K(Y, beta) then multiplies pointwise by K_X   and sums over basis :   sum_{b} K_X(X,X_b) \partial_Y \partial_basis K(Y, basis_b)*beta_b
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# beta 		: M by d matrix of weights
		# K_X 		: N by M matrix
		# output    : N by d vector   
		raise NotImplementedError()
	
	def cross_laplacian(self, Y, basis, K ):
		# computes Laplacian of laplacian(self, Y, basis,  K_X ) with respect to Y_n and sums over n
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# K_X 		: N by M matrix
		# output    : scalar  
		raise NotImplementedError()

	def quad_part(self, Y,beta,lmbda, K):
		raise NotImplementedError()


	def __add__(self, other):
		if other.D != self.D:
			raise NameError('Dimensions of kernels do not match !')
		else:
			new_kernel = CombinedKernel(self.D, [self, other])
			return new_kernel


class CombinedKernel(BaseKernel):
	def __init__(self,D,  kernels):
		BaseKernel.__init__(self, D)
		self.kernels = kernels

	def kernel(self, X,basis=None):
		K = 0
		for kernel in self.kernels:
			K += kernel.kernel(X, basis=basis)
		return K 

	def update_kernel(self,Y, basis, K):
		raise NotImplementedError()


	def hessian(self,Y, K):
		H = 0
		for kernel in self.kernels:
			H += kernel.hessian(Y, K)
		return H


	def w_grad(self, Y, basis,  beta , K ):
		h  = 0 
		for kernel in self.kernels:
			h += kernel.w_grad(Y, basis, beta,  K)
		return h

	
	def laplacian(self, Y, basis,  K ):
		h = 0
		for kernel in self.kernels:
			h += kernel.laplacian(Y, basis,  K)
		return h

	
	def grad_laplacian(self,Y, basis, K ):
		h = 0
		for kernel in self.kernels:
			h += kernel.grad_laplacian(Y, basis,  K)
		return h
		
	
	def w_cross_hessian(self, Y, basis, beta,  K):
		h = 0
		for kernel in self.kernels:
			h += kernel.w_cross_hessian(Y, basis, beta, K)
		return h

	
	def cross_laplacian(self, Y, basis, K):
		h = 0
		for kernel in self.kernels:
			h += kernel.cross_laplacian(Y, basis, K)
		return h

	def quad_part(self, Y,beta,lmbda, K):

		h = 0
		for kernel in self.kernels:
			h += kernel.quad_part(Y, beta,lmbda, K)
		return h


class NullKernel(BaseKernel):
	def __init__(self):
		BaseKernel.__init__(self,1)
		self.params =1.
		self.isNull = True

	def set_params(self,params):
		pass
	def get_params(self):
		return self.params

	def kernel(self, X, basis = None):
		return self._kernel(self.params, X, basis)
	def _kernel(self,param, X, basis = None):
		if basis is None:
			basis = X
		N_X,_ = X.shape
		N_basis ,_ = basis.shape 
		return np.ones([N_X, N_basis])













