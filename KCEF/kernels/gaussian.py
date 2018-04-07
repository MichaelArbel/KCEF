import autograd.numpy as np
from KCEF.kernels.base_kernel import BaseKernel


class Gaussian(BaseKernel):
	def __init__(self, D,  sigma):
		BaseKernel.__init__(self, D)
		self.params 	= 1./np.array(sigma)

 	def set_params(self,params):
 		# stores the inverse of the bandwidth of the kernel
		self.params = 1./params

	def get_params(self):
		# returns the bandwidth of the gaussian kernel
		return 1./self.params


	def square_dist(self, X, basis = None):
		# Squared distance matrix of pariwise elements in X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._square_dist( X, basis = None)

	def kernel(self, X,basis=None):

		# Gramm matrix between vectors X and basis
		# Inputs:
		# X 	: N by d matrix of data points
		# basis : M by d matrix of basis points
		# output: N by M matrix

		return self._kernel(self.params,X, basis)

	def weighted_kernel(self, Y, basis, K):
		# Computes the Hadamard product between the gramm matrix of (Y, basis) and the matrix K
		# Input:
		# Y 	: N by d matrix of data points
		# basis : M by d matrix of basis points		
		# K 	: N by M matrix
		# output: N by M matrix   K had_prod Gramm(Y, basis)

		return self._weighted_kernel(self.params, Y,basis, K)

	
	def hessian_bloc_dim(self,Y_i,Y_j,K, i, j):
		# Partial weighted Hessian of k(Y_1, Y_2) output = K_X(X_i, X_j) kro_prod  \partial_i \partial_i k(Y_i,Y_j) for fixed i and j
		# Inputs
		# Y_i, Y_j  : N by 1  slices of data points on dimensions i and j , N is the number of data points
		# K_X 		: N by N matrix
		# output    : N by N matrix 

		return self._hessian_bloc_dim(self.params,Y_i,Y_j,K, i, j)

	def hessian(self,Y, K_X):

		# weighted hessian of   k(Y_1, Y_2)  : output = K_X(X_i, X_j) kro_prod  \partial_i \partial_i k(Y_i,Y_j) for  1 < i, j <d 
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# K_X 		: N by N matrix
		# output    : Nd by Nd matrix   

		return self._hessian(self.params,Y, K_X)

	def w_grad(self, Y, basis,  beta , K_X ):
		# computes grad weighted_kernel(Y, basis, K_X) with respect to basis, multiplies by beta and sums over basis:  sum_b grad weighted_kernel(Y, basis_b, K_X) beta_b
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# beta 		: M by d matrix of weights
		# K_X 		: N by M matrix
		# output    : N by 1 vector   

		return self._w_grad(self.params, Y, basis,  beta , K_X )


	def laplacian(self, Y, basis,  K_X ):
		
		# Computes the Laplacian of weighted_kernel(Y, basis, K) with respect to basis and sum over basis:    \sum_{basis}   Delta_{basis} weighted_kernel(Y, basis, K)      
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# K_X 		: N by M matrix
		# output    : N by 1 vector   		

		return self._laplacian(self.params, Y, basis,  K_X )

	def grad_laplacian(self,Y, basis, K_X):
		# Computes the jacobian with respect to Y of laplacian(Y, basis, K_X) :   grad_Y laplacian(Y, basis, K_X)
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# K_X 		: N by M matrix
		# output    : N by d vector   


		return self._grad_laplacian(self.params,Y, basis, K_X)

	def w_cross_hessian(self, Y, basis, beta,  K_X):
		# Computes the product of beta_n with  \partial_Y\partial_beta K(Y, beta) then multiplies pointwise by K_X   and sums over basis :   sum_{b} K_X(X,X_b) \partial_Y \partial_basis K(Y, basis_b)*beta_b
		# Inputs
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# beta 		: M by d matrix of weights
		# K_X 		: N by M matrix
		# output    : N by d vector   

		return self._w_cross_hessian(self.params, Y, basis, beta,  K_X)


	def cross_laplacian(self, Y, basis, K_X ):
		# computes Laplacian of laplacian(self, Y, basis,  K_X ) with respect to Y_n and sums over n
		# Y 		: N by d matrix of data points, N is the number of points and d the dimension
		# basis 	: M by d matrix of basis points
		# K_X 		: N by M matrix
		# output    : scalar  		

		return self._cross_laplacian(self.params,Y,basis, K_X)




	def quad_part(self, Y,beta,lmbda, K):
		N, d = Y.shape
		if K is None:
			K = np.ones([1, N])
		return np.zeros( [d, d, K.shape[0]])



# Private functions 

	def _square_dist(self,X, basis = None):
		if basis is None:
			n,d = X.shape
			dist = np.matmul(X,X.T)
			diag_dist = np.outer(np.diagonal(dist), np.ones([1, n]))
			dist = diag_dist + diag_dist.T - 2*dist
		else:
			n_x,d = X.shape
			n_y,d = basis.shape
			dist = -2*np.matmul(X,basis.T) + np.outer(np.sum(np.square(X), axis=1), np.ones([1, n_y])) + np.outer( np.ones([n_x,1]),np.sum(np.square(basis), axis=1))
		return dist 

	def _kernel(self,sigma,X,basis):
		dist = self._square_dist( X, basis = basis)
		return  np.exp(-dist*sigma)

	def _weighted_kernel(self,sigma, Y, basis, K):
		K2 = self._kernel(sigma,Y, basis =  basis)
		if K is None:
			return  K2
		else:
			return np.multiply(K, K2)

	def _hessian_bloc_dim(self,sigma,Y_i,Y_j,K, i, j):
		n = Y_i.shape[0]
		Y_ii = np.reshape(Y_i, [1,-1])
		Y_jj = np.reshape(Y_j, [1,-1])
		diff_i = np.tile(Y_ii, [n,1])
		diff_i = diff_i.T - diff_i
		diff_j = np.tile(Y_jj, [n,1])
		diff_j = diff_j.T - diff_j

		if i ==j :
			return (np.multiply(K, (2.*(sigma) - 4.*(sigma**2)*np.multiply(diff_i,diff_j) )))
		else:
			return - 4.*(sigma**2)*(np.multiply(K, np.multiply(diff_i,diff_j) ))

	def _hessian(self,sigma,Y, K_X):
		by_dim= False
		n_y, d = Y.shape
		K 		= self._weighted_kernel(sigma,Y, None , K_X)
		hessian = np.zeros((d*d,n_y , n_y ))
		Y = np.array(Y, order='F')
		for i in range(d):
			for j in range(d):
				c_start, c_end = j * n_y, j * n_y + n_y
				r_start, r_end = i * n_y, i * n_y + n_y
				tmp = self._hessian_bloc_dim(sigma,Y[:,i], Y[:,j], K, i, j)
				tmp = np.reshape(tmp,[1,tmp.shape[0], tmp.shape[0]])
				if i ==0 and j ==0:	
					hessian = 1.*tmp
				else:
					hessian = np.concatenate([hessian, tmp], axis = 0)
		hessian = np.reshape(hessian, [d,d,n_y,n_y])
		hessian = np.swapaxes(hessian,0,2)
		hessian = np.swapaxes(hessian,2,3)
		hessian = np.reshape(hessian, [d*n_y,d*n_y])

		return hessian




	def _w_grad(self,sigma, Y, basis,  beta , K_X ):

		n_y, d_y    = Y.shape
		n_basis,_   = basis.shape

		K = self._weighted_kernel(sigma,Y, basis , K_X)
	
		b_d = np.matmul(Y, beta.T) - np.outer( np.ones([n_y, 1]) , np.sum(np.multiply(beta, basis), axis=1) ) 
		K_b_mat = np.multiply(K, b_d)
		K_b_d 	= np.sum(K_b_mat, axis=1)
		return (2.*sigma)*K_b_d




	def _laplacian(self,sigma, Y, basis,  K_X ):

		n_y, d    = Y.shape
		n_basis,_   = basis.shape
		dist 	 = self._square_dist(Y, basis=basis)
		K 		 = self._weighted_kernel(sigma,Y, basis  , K_X)

		KK  = np.sum(K, axis=1)
		K_d_mat = np.multiply(K, dist)
		K_d = np.sum(K_d_mat, axis=1)

		return -(2.*sigma)*(d * KK - (2.*sigma)*K_d)



	def _grad_laplacian(self,sigma,Y, basis, K_X):

		dist 	= self._square_dist(Y, basis=basis)

		K 		= self._weighted_kernel(sigma,Y, basis , K_X)

		if basis is None:
			basis_Y =  Y
		else: 
			basis_Y = basis

		_, d = Y.shape
		K_d_mat = np.multiply(K, dist)
		G  =   4.*(sigma**2) *((2+d)*K - 2.*sigma * K_d_mat)
		K_d = np.sum(K_d_mat, axis=1)
		KK = np.sum(K, axis=1)

		tmp = 4.*(sigma**2) *((2+d)*KK - 2.*sigma * K_d)
		tmp = tmp.reshape([-1,1])
		h =   (np.multiply(tmp, Y) - np.matmul(G, basis_Y))
		return h


	def _w_cross_hessian(self,sigma, Y, basis, beta,  K_X):
		
		if beta is None:
			return 0
		else:

			K = self._weighted_kernel(sigma,Y, basis ,K_X)
			if basis is None:
				basis_Y = Y
			else:
				basis_Y = basis


			n_y, d_y    = Y.shape
			n_basis,_   = basis_Y.shape
			K_b 	= np.matmul(K, beta)
			b_d = np.matmul(Y, beta.T) - np.outer( np.ones([n_y, 1]) , np.sum(np.multiply(beta, basis_Y), axis=1) ) 

			K_b_mat = np.multiply(K, b_d)
			K_b_d 	= np.sum(K_b_mat, axis=1)

			K_b_y = np.matmul(K_b_mat, basis_Y)

			h = (2.*sigma)*K_b + (2.*sigma)**2 * (K_b_y - np.multiply(np.reshape(K_b_d, [-1,1]) , Y) )
			return h


	def _cross_laplacian(self,sigma, Y, basis, K_X ):

		_, D = Y.shape
		dist 	= self._square_dist(Y, basis=basis)
		K 		= self._weighted_kernel(sigma,Y, basis  , K_X)

		s_K 	= np.sum(K)
		K_d_mat = np.multiply(K, dist)
		K_d 	= np.sum(K_d_mat, axis=1)

		s_K_d 	= np.sum(K_d) 

		s_K_d_2     = np.tensordot(K_d_mat, dist)

		h = (2.*sigma)**2 * ( (2.*D + D**2 ) * s_K -  4.*(2.+D)*sigma *  s_K_d  + (2.*sigma)**2 * s_K_d_2  )

		return h

































