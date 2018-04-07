import autograd.numpy as np
from KCEF.kernels.base_kernel import BaseKernel



class MultivariateGaussian(BaseKernel):
	def __init__(self, D,  sigma):
		if D == 0:
			D=1
		BaseKernel.__init__(self, D)
		self.params 	= 1./(np.array(sigma))*np.ones([1,D])
 	def set_params(self,params):
		self.params = 1./params

	def get_params(self):
		return 1./self.params


	def square_dist(self, X, basis = None):
		return self._square_dist( X, basis = None)

	def kernel(self, X,basis=None):
		return self._kernel(self.params,X, basis)


	def update_kernel(self, Y, basis, K):
		return self._update_kernel(self.params, Y,basis, K)

	
	def hessian_bloc_dim(self,Y_i,Y_j,K, i, j):
		return self._hessian_bloc_dim(self.params,Y_i,Y_j,K, i, j)

	def hessian(self,Y, K_X):
		return self._hessian(self.params,Y, K_X)

	def w_grad(self, Y, basis,  beta , K_X ):
		return self._w_grad(self.params, Y, basis,  beta , K_X )


	def laplacian(self, Y, basis,  K_X ):
		return self._laplacian(self.params, Y, basis,  K_X )

	def grad_laplacian(self,Y, basis, K_X):
		return self._grad_laplacian(self.params,Y, basis, K_X)

	def w_cross_hessian(self, Y, basis, beta,  K_X):
		return self._w_cross_hessian(self.params, Y, basis, beta,  K_X)


	def cross_laplacian(self, Y, basis, K_X ):
		return self._cross_laplacian(self.params,Y,basis, K_X)




	def quad_part(self, Y,beta,lmbda, K):
		N, d = Y.shape
		if K is None:
			K = np.ones([1, N])
		return np.zeros( [d, d, K.shape[0]])



# Private functions 

	def _square_dist( self,X, basis = None):
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
		if basis is None:
			basis = X

		n_x,d = X.shape
		n_y,d = basis.shape
		#dist = -2*np.matmul(np.multiply(X, 1./sigma),basis.T) + np.outer(np.matmul(np.square(X), (1./sigma).T), np.ones([1, n_y])) + np.outer( np.ones([n_x,1]),np.matmul(np.square(basis),(1./sigma).T))		
		dist = -2*np.matmul(np.multiply(X, sigma),basis.T) + np.outer(np.reshape(np.matmul(np.square(X), sigma.T),[-1,1]), np.ones([1, n_y])) + np.outer( np.ones([n_x,1]),np.reshape(np.matmul(np.square(basis),sigma.T), [1,-1])	)	
		return  np.exp(-dist)

	def _update_kernel(self,sigma, Y, basis, K):
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
			return (np.multiply(K, (2.*sigma - 4.*(sigma**2)*np.multiply(diff_i,diff_j) )))
		else:
			return - 4.*(sigma**2)*(np.multiply(K, np.multiply(diff_i,diff_j) ))

	def _hessian(self,sigma,Y, K_X):
		by_dim= False
		n_y, d = Y.shape
		K 		= self._update_kernel(sigma,Y, None , K_X)
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

		K = self._update_kernel(sigma,Y, basis , K_X)
		

	#		if b_d is None:
		b_d = np.matmul(Y, beta.T) - np.outer( np.ones([n_y, 1]) , np.sum(np.multiply(beta, basis), axis=1) ) 
	#		if K_b_mat is None:
		K_b_mat = np.multiply(K, b_d)
	#		if K_b_d is None:
		K_b_d 	= np.sum(K_b_mat, axis=1)
		return (2.*sigma)*K_b_d




	def _laplacian(self,sigma, Y, basis,  K_X ):

		n_y, d    = Y.shape
		n_basis,_   = basis.shape
		dist 	 = self._square_dist(Y, basis=basis)
		K 		 = self._update_kernel(sigma,Y, basis  , K_X)

	#		if KK is None:
		KK  = np.sum(K, axis=1)
	#		if K_d_mat is None:
		K_d_mat = np.multiply(K, dist)
	#		if K_d is None:
		K_d = np.sum(K_d_mat, axis=1)

		return -(2.*sigma)*(d * KK - (2.*sigma)*K_d)



	def _grad_laplacian(self,sigma,Y, basis, K_X):

		dist 	= self._square_dist(Y, basis=basis)

		K 		= self._update_kernel(sigma,Y, basis , K_X)

		if basis is None:
			basis_Y =  Y
		else: 
			basis_Y = basis

		_, d = Y.shape
	#		if K_d_mat is None:
		K_d_mat = np.multiply(K, dist)
		G  =   4.*(sigma**2)*((2+d)*K - 2.*sigma * K_d_mat)

	#		if K_d is None:
		K_d = np.sum(K_d_mat, axis=1)
	#		if self_KK is None:
		KK = np.sum(K, axis=1)

		tmp =  4.*(sigma**2)*((2+d)*KK - 2.*sigma * K_d)
		tmp = tmp.reshape([-1,1])
		h = np.multiply(tmp, Y) - np.matmul(G, basis_Y)
		return h


	def _w_cross_hessian(self,sigma, Y, basis, beta,  K_X):
		
		if beta is None:
			return 0
		else:

			K = self._update_kernel(sigma,Y, basis ,K_X)
			if basis is None:
				basis_Y = Y
			else:
				basis_Y = basis


			n_y, d_y    = Y.shape
			n_basis,_   = basis_Y.shape
	#			if K_b is None:
			K_b 	= np.matmul(K, beta)
	#			if b_d is None:
			b_d = np.matmul(Y, beta.T) - np.outer( np.ones([n_y, 1]) , np.sum(np.multiply(beta, basis_Y), axis=1) ) 

	#			if K_b_mat is None:
			K_b_mat = np.multiply(K, b_d)
	#			if self_K_b_d is None:
			K_b_d 	= np.sum(K_b_mat, axis=1)

			K_b_y = np.matmul(K_b_mat, basis_Y)

			h = (2.*sigma)*K_b + (2.*sigma)**2 * (K_b_y - np.multiply(np.reshape(K_b_d, [-1,1]) , Y) )
			return h


	def _cross_laplacian(self,sigma, Y, basis, K_X ):

		_, D = Y.shape
		dist 	= self._square_dist(Y, basis=basis)
		K 		= self._update_kernel(sigma,Y, basis  , K_X)

		s_K 	= np.sum(K)
	#		if K_d_mat is None:
		K_d_mat = np.multiply(K, dist)
	#		if K_d is None:
		K_d 	= np.sum(K_d_mat, axis=1)

		s_K_d 	= np.sum(K_d) 

		s_K_d_2     = np.tensordot(K_d_mat, dist)

		h = (2.*sigma)**2 * ( (2.*D + D**2 ) * s_K -  4.*(2.+D)*sigma *  s_K_d  + (2.*sigma)**2 * s_K_d_2  )

		return h

































