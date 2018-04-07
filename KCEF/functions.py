import autograd.numpy as np
import scipy.linalg as lg
from scipy.stats import multivariate_normal
from autograd import value_and_grad
from autograd import grad as automatic_grad
from scipy.optimize import minimize
from tools import create_splits






def compute_h(kernel,base_density,  Y, K_X):
	# computes the vector h for the regression problem  GX = h
	params = np.reshape(kernel.params,[-1])
	return _compute_h(params,kernel,base_density,  Y, K_X)

def compute_G(kernel, Y, K_X  ):
	# computes the hessian G for the regression problem GX = h
	params = np.reshape(kernel.params,[-1])
	return _compute_G(params,kernel, Y, K_X  )


def score(kernel_x, kernel_y, base_density, X, Y, basis_X, basis_Y, beta, lmbda):
	# computes the score up to a constant. Please refer to the paper for more details about the equations.
	kernel_x_params = np.reshape(kernel_x.params,[-1])
	kernel_y_params = np.reshape(kernel_y.params,[-1])
	return _score(kernel_x, kernel_y, kernel_x_params, kernel_y_params, base_density, X, Y, basis_X, basis_Y, beta, lmbda)



def optimize_cv_score(kernel_x, kernel_y, base_density, X,Y, lmbda, n_split = 10, with_score= False):

	# returns optimized hyperparameters over cross-validated score

	n_total, d = Y.shape
	split = create_splits(Y, n_split)
	if kernel_x.isNull:
		X = np.zeros([n_total, 1])

	kernel_x_params = np.reshape(kernel_x.params,[-1])
	kernel_y_params = np.reshape(kernel_y.params,[-1])
	lmbda_param 	= np.reshape(lmbda,[-1])

	d_x = len(kernel_x_params)
	d_y = len(kernel_y_params)
	init_params = np.concatenate([kernel_x_params,kernel_y_params,lmbda_param])
	betas = _cv_beta(kernel_x, kernel_y,kernel_x_params, kernel_y_params, base_density, X,Y, lmbda, split)
	bounds = ()
	for i in range(d_x+d_y+1):
		bounds += ((1e-10, 10.),)

	options = { 'disp':True , 'maxls':20,  'gtol': 1e-4,'ftol': 1e-4,  'maxiter': 50,  'eps': 1e-08,'maxcor': 10, 'maxfun': 15000}
	
	def objective(params):

		kernel_x_params = params[:d_x]
		kernel_y_params = params[d_x:d_x+d_y]
		lmbda 			= params[-1]
		kernel_x.params = kernel_x_params
		kernel_y.params = kernel_y_params

		betas = _cv_beta(kernel_x, kernel_y, kernel_x_params,kernel_y_params,base_density, X,Y, lmbda, split )
		objective = np.mean(np.array([_score(kernel_x,kernel_y,kernel_x_params, kernel_y_params, base_density, X[idx[1],:], Y[idx[1],:], X[idx[0],:], Y[idx[0],:], betas[i,:,:], lmbda) for i,idx in enumerate(split)]))
		# Optional: can add an l2 term to regularize the objective
		objective += 0.1*np.sum(params**2)
		return objective 

	res = minimize(value_and_grad(objective), init_params, jac=True,bounds = bounds,
                          method='L-BFGS-B', options =options)

	params 			= res['x']
	kernel_x_params = params[:d_x]
	kernel_y_params = params[d_x:d_x+d_y]
	lmbda 			= params[-1]
	kernel_x.params = kernel_x_params
	kernel_y.params = kernel_y_params

	cv = objective(params)

	kernel_x_params = kernel_x.get_params()
	kernel_y_params = kernel_y.get_params()
	lmbda 			= params[-1]

	CV_score 		= res['fun']
	message 		= res['message']
	
	if with_score:
		return kernel_x_params, kernel_y_params, lmbda, CV_score,message
	else:
		return kernel_x_params, kernel_y_params, lmbda


def cv_score(kernel_x, kernel_y, base_density, X,Y, lmbda, n_split = 10):
	n_total, d = Y.shape
	split = create_splits(Y, n_split)
	if kernel_x.isNull:
		X = np.zeros([n_total, 1])

	fit_metric = []
	kernel_x_params = np.reshape(kernel_x.params,[-1])
	kernel_y_params = np.reshape(kernel_y.params,[-1])
	betas = _cv_beta(kernel_x, kernel_y,kernel_x_params,kernel_y_params,  base_density, X,Y, lmbda, split)
	for i , train_idx, test_idx in enumerate(split):
		beta = betas[i,:,:]
		fit_metric.append(score(kernel_x, kernel_y, base_density, X[test_idx, :], Y[test_idx, :], X[train_idx, :], Y[train_idx,:], beta, lmbda))
	return np.mean(np.array(fit_metric))



def fit( kernel_x, kernel_y, base_density, X, Y, lmbda):
	n_y, d_y = Y.shape
	K_X = kernel_x.kernel(X)
	
	h = compute_h(kernel_y,base_density, Y, K_X)
	G = compute_G(kernel_y, Y, K_X)
	
	np.fill_diagonal(G, np.diag(G) + n_y * lmbda)
	cho_lower = lg.cho_factor(G)
	beta = lg.cho_solve(cho_lower, h / lmbda)    
	return beta.reshape(n_y,d_y)



def log_partition(kernel_x, kernel_y, base_density,X, Y, base_x, base_y, beta,lmbda, num_samples = 1000):

	# Data need to be centered and normalized 

	d_node = Y.shape[1]
	sigmas	= 4.
	mu 		= np.mean(Y, axis=0)	

	num_samples = 1000
	num_x 	= X.shape[0]
	X_rep 	= np.repeat(X, num_samples, axis = 0)
	tmp_sigmas = sigmas*np.ones([X_rep.shape[0], 1])

	samples = np.random.multivariate_normal(mu, np.eye(mu.shape[0]) ,  num_samples * num_x )
	samples = np.multiply(samples, tmp_sigmas)
	chunk_size = 10000
	if num_samples * num_x > 50000:
		log_diff = np.zeros([num_samples*num_x])
		num_chunks = 2000

		chunk_size = num_samples * num_x/num_chunks
		for i in range(num_chunks):
			log_diff[i*chunk_size:(i+1)*chunk_size] = log_pdf(kernel_x, kernel_y, base_density,  X_rep[i*chunk_size:(i+1)*chunk_size], samples[i*chunk_size:(i+1)*chunk_size], base_x, base_y, beta, lmbda)
	
	else:
		log_diff = log_pdf(kernel_x, kernel_y, base_density,  X_rep, samples, base_x, base_y, beta, lmbda)
	# computing proposal log-pdf
	tmp = multivariate_normal.logpdf(samples, mean=mu) + (d_node/2.)*np.log(2*np.pi)
	tmp = np.reshape(tmp, [-1, 1])
	tmp = np.multiply(tmp, 1./tmp_sigmas**2)
	tmp = tmp - d_node*(np.log(2*np.pi)/2. +  np.log(tmp_sigmas))
	tmp = np.reshape(tmp, [-1])
	# substracting proposal log-pdf
	log_diff -= tmp
	if base_x.shape[0]>0:
		log_diff = np.reshape(log_diff, [-1, num_samples])
		max_diff = np.max(log_diff, axis=1)
		log_diff -= np.reshape(max_diff, [-1,1])
		shifted_log_Z 	= np.log( np.mean( np.exp(log_diff) , axis=1)) 

		log_Z 	= max_diff + shifted_log_Z

		shifted_log_Z_2  		= np.log(np.mean(np.exp(2*log_diff), axis=1))

	else:
		log_diff = np.reshape(log_diff, [-1, 1])
		max_diff = np.max(log_diff)
		log_diff -= max_diff

		shifted_log_Z 	= np.log( np.mean( np.exp(log_diff))) 

		log_Z 	= max_diff + shifted_log_Z

		shifted_log_Z_2  		= np.log(np.mean(np.exp(2*log_diff)))

	std_log_Z 	= shifted_log_Z_2 - 2* shifted_log_Z
	std_log_Z 	= np.sqrt((np.exp(std_log_Z)-1)/num_samples)

	if base_x.shape[0]==0:
		log_Z 		= log_Z*np.ones(Y.shape[0])
		std_log_Z 	= std_log_Z*np.ones(Y.shape[0])

	return log_Z, std_log_Z

def log_pdf(kernel_x, kernel_y, base_density,  X, Y, basis_X, basis_Y, beta, lmbda):
	n_y, d_y    = Y.shape
	n_basis,_   = basis_Y.shape

	K = kernel_x.kernel(X, basis_X)
		
	log_q_basis = base_density.log_pdf(Y)
	grad_log_q_basis = base_density.grad(basis_Y)

	log_pdf 	= kernel_y.laplacian(Y, basis_Y, K)
	if grad_log_q_basis is not None:
		log_pdf += kernel_y.w_grad(Y, basis_Y, grad_log_q_basis ,K)
	log_pdf 	= -1./(lmbda * n_basis) * log_pdf 

	log_pdf 	+= log_q_basis  + kernel_y.w_grad(Y, basis_Y, beta ,K) 

	return log_pdf

def grad(kernel_x, kernel_y, base_density,X, Y, basis_X, basis_Y, beta, lmbda):

	n_y, d_y    = Y.shape
	n_basis,_   = basis_Y.shape
	K = kernel_x.kernel(X, basis_X)
	grad_log_q_basis = base_density.grad(basis_Y)
	grad_xi  	=  kernel_y.grad_laplacian(Y, basis_Y,K)
	if grad_log_q_basis is not None:  
		grad_xi		+=   kernel_y.w_cross_hessian(Y, basis_Y, grad_log_q_basis, K)
		
	grad_xi 	=	-1./(lmbda *n_basis) * grad_xi

	grad  		=  grad_xi + kernel_y.w_cross_hessian(Y, basis_Y, beta, K)

	return grad




def _compute_h(params,kernel,base_density,  Y, K_X):
	# private function
	# computes the vector h for the regression problem  GX = h
	n, d = Y.shape
	h = kernel._grad_laplacian(params,Y,Y,  K_X)
	grad_log_q = base_density.grad(Y)
	if grad_log_q is not None:  
		h		+=   kernel._w_cross_hessian(params,Y, Y, grad_log_q, K_X)
	return h.reshape(-1)/n

def _compute_G(params,kernel, Y, K_X  ):
	# private function
	# computes the hessian G for the regression problem GX = h
	return kernel._hessian(params,Y, K_X)

def _score(kernel_x, kernel_y, kernel_x_params, kernel_y_params, base_density, X, Y, basis_X, basis_Y, beta, lmbda):
	# computes the score up to a constant. Please refer to the paper for more details about the equations.
	n_y, d_y    = Y.shape
	n_basis,_   = basis_Y.shape
	K 				= kernel_x._kernel(kernel_x_params,X, basis_X)
	grad_log_q_Y 		= base_density.grad(Y)
	grad_log_q_basis 	= base_density.grad(basis_Y)
	xi_dot 		 		= kernel_y._cross_laplacian(kernel_y_params,Y, basis_Y, K) + np.tensordot(kernel_y._grad_laplacian(kernel_y_params,basis_Y,Y, K.T ), grad_log_q_basis )
	T_xi_dot 	 		= np.tensordot(kernel_y._grad_laplacian(kernel_y_params,basis_Y,Y, K.T ), beta )
	hess_diag_f_hat 	= T_xi_dot   - xi_dot/(lmbda * n_basis)
	grad_f_hat  		= -1./(lmbda *n_basis) * ( kernel_y._grad_laplacian(kernel_y_params,Y, basis_Y,K)  + kernel_y._w_cross_hessian(kernel_y_params,Y, basis_Y, grad_log_q_basis, K) ) \
							+ kernel_y._w_cross_hessian(kernel_y_params,Y, basis_Y, beta, K)
	cross_grad_f_hat 	= np.sum(np.multiply(grad_log_q_Y, grad_f_hat))
	score 	 		= (1.0/n_y)*( 0.5* np.sum(np.square(grad_f_hat))  + cross_grad_f_hat  + hess_diag_f_hat)

	return score


def _cv_beta(kernel_x, kernel_y,kernel_x_params,kernel_y_params,  base_density, X,Y, lmbda,  split ):
	
	# fits on k-1 folds of the training dataset and repeats the operation. The output is a tensor beta of shape:
	# beta : k times N times d
	# k is the number of folds, N the number of data point in the k-1 folds and   d is the dimension of the data.

	n_total, d = Y.shape
	n_train = split[0][0].shape[0]
	n_test 	= split[0][1].shape[0]
	K_X 				= kernel_x._kernel(kernel_x_params,X,X)

	G = _compute_G(kernel_y_params,kernel_y, Y, K_X)   + n_train * lmbda*np.eye(n_total)

	G = np.linalg.inv(G)
	
	num_folds = 0
	for train_idx, test_idx in split:
		train_idx_tot = d*np.repeat(train_idx, d) + np.tile(np.array(range(d)), train_idx.shape[0])
		test_idx_tot = d*np.repeat(test_idx, d) + np.tile(np.array(range(d)), test_idx.shape[0])


		h = _compute_h(kernel_y_params,kernel_y,base_density, Y[train_idx,:], K_X[np.ix_(train_idx, train_idx)])
		GG = _compute_G(kernel_y_params,kernel_y, Y[train_idx,:], K_X[np.ix_(train_idx, train_idx)])

		beta = np.matmul(G[np.ix_(train_idx_tot, train_idx_tot)], h)

		h = np.matmul(G[np.ix_(test_idx_tot,train_idx_tot)], h) 	

		beta_tmp = np.linalg.solve(G[np.ix_(test_idx_tot, test_idx_tot)], h)
		beta -= np.matmul( G[np.ix_(train_idx_tot, test_idx_tot)] , beta_tmp)
		beta = beta/lmbda
		beta = np.reshape(beta, [1,-1,d])
		if num_folds == 0:
			betas = 1.*beta
		else:
			betas = np.concatenate([betas, beta], axis = 0)
		num_folds += 1

	return betas

