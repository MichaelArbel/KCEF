import autograd.numpy as np

from KCEF.densities.gaussian import IsotropicCenteredGaussian
from KCEF.kernels.gaussian import Gaussian
from KCEF.kernels.base_kernel import NullKernel

from KCEF.functions import  log_pdf, log_partition, grad, score, cv_score,optimize_cv_score, fit
from KCEF.tools import make_graph
from KCEF.sampler.hmc import HMCBase
from KCEF.sampler.ancestral_hmc import ancestral_hmc

class KCEF(object):
	def __init__(self, graph, lmbda, kernels, base_density = None, basis=None):
		# graph     : grah for the conditional structure
		# sigma     : structure for the kernels
		# lmbda     : list of regularizers, one for each node in the graph 

		self.num_nodes = len(graph)
		self.kernels = kernels
		if base_density is None:
			base_density  = []
			for i in range(self.num_nodes):
				#base_density.append(FlatDensity())
				base_density.append(IsotropicCenteredGaussian())

		self.base_density = base_density
		self.lmbda = lmbda
		self.basis = basis
		self.graph = graph

		# initial RKHS function is flat
		self.beta = [0]*len(graph) 
		self.cur_node_idx 	= 0
		self.cur_x 		= np.reshape(np.array(0.), [1,1])
		self.cur_log_pdf= 0.
		self.cur_grad 	= 0.
		self.is_updated = False
		self.is_fitted = False
		self.samples_indices= None

	def split_cond(self, data, node):
		# splits data into two arrays Y and X along the dimension d of data. 
		Y = np.array([data[:,i] for i in node[0]]).T
		X = np.array([data[:,i] for i in node[1]]).T
		if len(node[1]) == 0:
			X = np.zeros([data.shape[0], 1])
		return X, Y

	def reset(self):
		self.is_updated = False

	def fit(self, data):
		if self.is_fitted:
			self.is_fitted = False
		self.basis = data
		for e, node in enumerate(self.graph):
			X,Y = self.split_cond(data, node)
			self.beta[e] = fit(self.kernels[e][1], self.kernels[e][0],self.base_density[e] , X, Y, self.lmbda[e])
		self.is_fitted = True

	def set_cond(self,x,node):

		if x is not None:
			if len(x.shape)<2:
				if len(x.shape) == 0:
					x = np.reshape(x, [1,-1])
				else:
					if x.shape[0] <=1:
						x = np.reshape(x, [1,-1])
					else:
						x = np.reshape(x, [-1,1])
			self.cur_x = x
		else:
			self.cur_x = np.reshape(0., [1,-1])
		self.cur_node_idx = node
		self.is_updated = False


	def compute(self, function,   y, x = None, node_idx = None):

		if node_idx is None:
			node_idx = self.cur_node_idx
		node = self.graph[node_idx]
		if x is None:
			x = self.cur_x


		y = np.array(y) 
		if len(y.shape)<2:
			if len(y.shape) ==0:
				y = np.reshape(y, [1,-1])
			else:
				if y.shape[0] <=1:
					y = np.reshape(y, [1,-1])
				else:
					y = np.reshape(y, [-1,1])
		basis_X,basis_Y = self.split_cond(self.basis, node)

		kernel_x = self.kernels[node_idx][1]
		kernel_y = self.kernels[node_idx][0]
		lmbda   = self.lmbda[node_idx]
		beta    = self.beta[node_idx]
		base_density = self.base_density[node_idx]

		out = function(kernel_x, kernel_y,base_density,  x, y, basis_X, basis_Y, beta, lmbda)

		return out
	def update_parameters(self, params):

		for e, node in enumerate(self.graph):
			kernel_x_params, kernel_y_params, lmbda = params[e]
			self.kernels[e][1].set_params(kernel_x_params)
			self.kernels[e][0].set_params(kernel_y_params)
			self.lmbda[e] 	 	=  lmbda


	def log_pdf(self, y, x = None, node = None ):
		self.cur_log_pdf = self.compute( log_pdf,  y, x , node)
		if len(np.array(self.cur_log_pdf).shape) == 0:
			self.cur_log_pdf = np.float(self.cur_log_pdf)
		return self.cur_log_pdf

	def log_partition(self, y, x=None, node=None):
		return self.compute( log_partition,  y, x , node)
		
	def grad(self,y, x =None, node = None, as_array = False ):
		self.cur_grad = self.compute( grad,  y, x , node)
		if as_array:
			return np.reshape(self.cur_grad, [-1])
		else:
			return self.cur_grad
	def score(self,y, x =None, node = None ):
		return self.compute( score,  y, x , node)

	def cv_score(self, data, n_split = 10, split = None):
		CV_score = 0.
		for e, node in enumerate(self.graph):
			X,Y = self.split_cond(data, node)
			CV_score += cv_score(self.kernels[e][1], self.kernels[e][0],self.base_density[e] , X, Y, self.lmbda[e], n_split, split)

		return CV_score

	def optimize_cv_score(self, data, n_split = 10, split = None):
		res = []
		for e, node in enumerate(self.graph):
			X,Y = self.split_cond(data, node)
			res.append(optimize_cv_score(self.kernels[e][1], self.kernels[e][0],self.base_density[e] , X, Y, self.lmbda[e], n_split, split))
		self.update_parameters(res)
		return res

	def total_likelihood(self, data):
		if self.is_fitted:
			total_likelihood = np.zeros([len(self.graph), data.shape[0]])
			total_std_log_Z = np.zeros([len(self.graph), data.shape[0]])
			for e, node in enumerate(self.graph):
				x,y = self.split_cond(data, node)
				log_Z,  std_log_Z = self.log_partition( y, x, e)
				cond_likelihood =  self.log_pdf( y, x, e)  - log_Z
				total_likelihood[e,:] = cond_likelihood
				total_std_log_Z[e,:] = std_log_Z

			likelihood = np.mean(np.sum(total_likelihood, axis= 0))
			std_likelihood = np.var(np.sum(total_likelihood, axis= 0)) + np.mean(np.sum(total_std_log_Z**2, axis= 0))
			std_likelihood = np.sqrt(std_likelihood)
			return likelihood, std_likelihood


	def sample(self,num_samples):
		if self.is_fitted:
			num_iter    = 20
			momentums = []
			for i in range(len(self.graph)):
				sigma_base = np.std(self.basis[:, self.graph[i][0]])
				momentums.append(IsotropicCenteredGaussian(D= len(self.graph[i][0]), sigma=sigma_base))  
			hmc = HMCBase(self, momentums[0], adaptation_schedule=None)
			samples = ancestral_hmc(hmc, momentums, self.graph, num_samples, num_iter)
			return samples
		else: 
			raise ValueError('the model is not fitted!! please first fit the model')

class KCEF_Gaussian(KCEF):
	def __init__(self, graph_type, d, sigma_x = 1., sigma_y = 1., lmbda = 0.0001, graph = None):
		if graph_type == 'custom':
			if graph is None:
				raise ValueError('Please provide a graph ex:  graph = [[[1],[0]]]')
		else:
			graph = make_graph(graph_type,d)
		lmbdas = []
		kernels = []
		for i in range(len(graph)):
			if len(graph[i][1]) == 0:
				kernel_x = NullKernel()
			else:  
				kernel_x = Gaussian(len(graph[i][1]), sigma_x)
			kernel_y = Gaussian(len(graph[i][0]), sigma_y)
			kernels.append([ kernel_y , kernel_x ])
			lmbdas.append(lmbda)
		KCEF.__init__(self, graph, lmbdas, kernels)




