# data generator
import numpy as np

class DataGenerator(object):
	def __init__(self):
		self.graph = 0
	def generate(self, N =-1):
		raise NotImplementedError()

	def generate_cond(self,X):
		raise NotImplementedError()


	def log_pdf_cond(self,data, node, X = None, compute_gradient=None, compute_score= None ):
		raise NotImplementedError()


class GaussianGenerator(DataGenerator):
	def __init__(self, mu, sigma ):
		DataGenerator.__init__(self)
		self.mu = 1*mu
		self.sigma = 1*sigma
		d_x 		= len(self.mu[0])
		d_y 		= len(self.mu[1]) - d_x

		self.graph = []
		self.graph.append([range(d_x), []])
		self.graph.append([range(d_x,d_y+d_x ), range(d_x)])

	def generate(self,N):
		d_x 		= len(self.mu[0])
		d_y 		= len(self.mu[1]) - d_x
		X         	= np.random.multivariate_normal(self.mu[0], self.sigma[0] ,  N)
		Y 			= self.generate_cond(X, N)
		data 		= np.concatenate([X,Y],axis= 1)
		return data

	def generate_cond(self, X , N):
		d_x 		= len(self.mu[0])
		d_y 		= len(self.mu[1]) - d_x

		Sigma_Y_X   = self.sigma[1][d_x:,d_x:] -  np.matmul(np.matmul(self.sigma[1][d_x:,0:d_x], np.linalg.inv(self.sigma[1][0:d_x,0:d_x])), self.sigma[1][0:d_x:,d_x:])
		mu_Y_X      = self.mu[1][d_x:] + np.matmul(np.matmul(self.sigma[1][d_x:,0:d_x], np.linalg.inv(self.sigma[1][0:d_x,0:d_x])), (X - self.mu[1][0:d_x]).T).T
		Y           = mu_Y_X + np.random.multivariate_normal([0,0], Sigma_Y_X ,  N)
		return Y

	def log_pdf_cond(self,data, node, X = None, compute_gradient=None, compute_score= None ):
		return 0


class MixtureGaussianGenerator(DataGenerator):
	def __init__(self, mu, sigma ):
		DataGenerator.__init__(self) 
		self.mu = 1*mu
		self.sigma = 1*sigma
		d_x 		= len(self.mu[0])
		d_y 		= len(self.mu[1]) - d_x

		self.graph = []
		self.graph.append([range(d_x), []])
		self.graph.append([range(d_x,d_y+d_x ), range(d_x)])

	def generate(self,N):
		d_x 		= len(self.mu[0])
		d_y 		= len(self.mu[1]) - d_x
		X         = np.random.multivariate_normal(self.mu[0], self.sigma[0] ,  N)

		Y 			= np.zeros([N,d_y])
		for i in range(N):
			Y[i, :] = self.generate_cond( X[i, :], 1 )
		data = np.concatenate([X,Y],axis= 1)
		return data

	def generate_cond(self, X, N ):
		d_x 		= len(self.mu[0])
		d_y 		= len(self.mu[1]) - d_x
		alpha 		= 0.1
		Sigma_Y_X_1   = alpha*(1 + np.linalg.norm(X) ) * np.eye(d_y,d_y)
		Sigma_Y_X_2   = alpha*(1 + np.linalg.norm(X) ) * np.eye(d_y,d_y)


		sign  		  = 2*(np.array(X)>0)-1
		sign 		  = np.reshape(sign, [-1, d_x])
		mu_Y_X_1      =  np.multiply(self.mu[1][0:d_x],sign) + 0.1*np.linalg.norm(X) 
		sign[:,-1] 	  = -sign[:,-1]
		mu_Y_X_2      =  np.multiply(self.mu[1][d_x:],sign) + 0.1*np.linalg.norm(X)
		pi = np.random.rand(N)
		Y_1 =  mu_Y_X_1 + np.random.multivariate_normal(np.zeros([d_y]), Sigma_Y_X_1 ,  N)
		Y_2  = mu_Y_X_2 + np.random.multivariate_normal(np.zeros([d_y]), Sigma_Y_X_2 ,  N)
		Y = np.multiply(np.outer((pi<0.5),np.ones([1,d_y]))  , Y_1) + np.multiply(np.outer((pi>=0.5),np.ones([1,d_y])) , Y_2) 
		return Y 

	def log_pdf_cond(self,data, node, X = None, compute_gradient=None, compute_score= None ):

		return 0



class MoonGenerator(DataGenerator):
	def __init__(self, R = 1.,alpha = 1. ):
		DataGenerator.__init__(self) 
		self.R = R 			# positive number
		self.alpha = alpha 	# positive number
		self.epsilon = 0.01*(self.alpha + 1)*self.R

	def generate(self,N):
		T      	= np.random.uniform( size = N)*((self.alpha+1)*self.R - self.epsilon) + self.epsilon
 		T 		= np.reshape(np.array(T), [-1, 1])
 		data  = self.generate_cond(T)
 		return np.concatenate([T, data],axis= 1)
	def generate_cond(self, T, N = None ):
		T = np.reshape(np.array(T), [-1, 1])
		if N is None:
			N = T.shape[0]
		U = np.reshape(np.random.uniform(size=N), [-1, 1])
		V = 0.5*(T + (1-self.alpha**2)*self.R**2/T  )
		X = np.multiply(U, self.R+V) - V
		# Generating Y coordinates
		A = np.sqrt(self.R**2 - X**2 )
		B = np.sqrt(np.maximum(0., (self.alpha*self.R)**2 - (X+T)**2))
		W = np.random.uniform(size=N)
		Y = np.multiply(np.reshape(W, [-1, 1]), A-B)+B
		W = np.random.uniform(size=N)
		W = (W<0.5)
		W = W- (1-W)
		Y = np.multiply(np.reshape(W, [-1, 1]),Y)

		X = np.reshape(X, [-1,1])
		Y = np.reshape(Y, [-1,1])
		return np.concatenate([X,Y],axis= 1)

	def log_pdf_cond(self,data, node, X = None, compute_gradient=None, compute_score= None ):

		return 0

class WaveGenerator(DataGenerator):
	def __init__(self, w_x , w_y):
		DataGenerator.__init__(self) 
		self.w_x = w_x
		self.w_y = w_y 			# positive number
		self.d = len(w_y)
		self.state = 0

	def generate(self,N):
		self.state = 0
		X      	= np.random.uniform( size = N)
 		X 		= np.reshape(np.array(X), [-1, 1])
 		X_tmp   = X
 		data 	= self.generate_cond(X) 
 		return np.concatenate([X,data], axis=1)

 	def generate_cond(self,X,N=None):
 		data = []
 		X_tmp   = X
  		for i in range(self.d):
 			self.state = i
 			Y = np.reshape(self.generate_cond_aux(X_tmp, N=N), [-1,1])
 			data.append(1*Y)
 			X_tmp = Y
 		data = np.concatenate(np.array(data), axis=1)
 		return data
		
	def generate_cond_aux(self, X, N = None ):
		X = np.array(X)
		sample = False
		if len(X.shape)>0:
			if X.shape[0] > 0:
				sample = True
		else:
			one_variable = True
			sample = True
		if sample:
			X = np.reshape(np.array(X), [-1, 1])
			if N is None:
				N = X.shape[0]
			else:
				if X.shape[0]!=N:
					X = X[0,0]*np.ones([N,1])
			Y = np.zeros([N,1])
			success = np.zeros([N,1])
			U = np.reshape(np.random.uniform(size=N), [-1, 1])
			acceptance = 0.5*(1 + np.multiply(np.sin(2*np.pi*self.w_y[self.state]*U), np.sin(2*np.pi*self.w_x[self.state]*X)))
			acceptance =acceptance/(1+ np.sin(2*np.pi*self.w_x[self.state]*X)*(1-np.cos(2*np.pi*self.w_y[self.state]))/(2*np.pi*self.w_y[self.state]) )
			acceptance = np.reshape(acceptance, [-1])
			V = np.random.uniform(size=N)
			success = (V<acceptance)
			Y[success] = U[success]
			Y[True ^ success] = self.generate_cond_aux(X[True ^ success])
			return Y

	def log_pdf_cond(self,data, node, X = None, compute_gradient=None, compute_score= None ):

		return 0


class FromFileGenerator(DataGenerator):
	def __init__(self, filename):
		DataGenerator.__init__(self) 
		self.filename = filename
		self.mean = 0.
		self.std = 1.
	def generate(self, N= -1, dataset_type= 'train', format = 'TVT'):
		alldata = np.load(self.filename)
		if format == 'TVT':
			if dataset_type == 'train':
				data = alldata[0]
				self.mean = np.mean(data, axis=0)
				self.std  = np.std(data, axis =0)
			elif dataset_type == 'validation':
				data = alldata[1]
			elif dataset_type == 'test':
				data = alldata[2]
		data =  (data - self.mean)/self.std
		N_total  = data.shape[0]
		if N >0:
			N_total = min(N,N_total)
		return data[:N,:]


