import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy

class AdversarialBIM(Strategy):
	def __init__(self, X, Y, unlabeled_x, net, handler, nclasses, args={}):

		"""
		Implementation of Adversial Bim Strategy.
		This class extends :class:`active_learning_strategies.strategy.Strategy`
		to include entropy sampling technique to select data points for active learning.

		Parameters
		----------
		X: numpy array
		    Present training/labeled data   
		y: numpy array
		    Labels of present training data
		unlabeled_x: numpy array
		    Data without labels
		net: class
		    Pytorch Model class
		handler: class
		    Data Handler, which can load data even without labels.
		nclasses: int
		    Number of unique target variables
		args: dict
		    Specify optional parameters
		    
		    batch_size 
		    Batch size to be used inside strategy class (int, optional)

		    eps
		    epsilon value for gradients
		"""
		
		if 'eps' in args:
			self.eps = args['eps']
		else:
			self.eps = 0.05
		
		super(AdversarialBIM, self).__init__(X, Y, unlabeled_x, net, handler, nclasses, args={})

	def cal_dis(self, x):
		nx = torch.unsqueeze(x, 0)
		nx.requires_grad_()
		eta = torch.zeros(nx.shape)

		out = self.model(nx+eta)
		py = out.max(1)[1]
		ny = out.max(1)[1]
		while py.item() == ny.item():
			loss = F.cross_entropy(out, ny)
			loss.backward()

			eta += self.eps * torch.sign(nx.grad.data)
			nx.grad.data.zero_()

			out = self.model(nx+eta)
			py = out.max(1)[1]

		return (eta*eta).sum()

	def select(self, budget):
		"""
		Select next set of points

		Parameters
		----------
		budget: int
		    Number of indexes to be returned for next set

		Returns
		----------
		idxs: list
		    List of selected data point indexes with respect to unlabeled_x
		"""	
		self.model.cpu()
		self.model.eval()
		dis = np.zeros(self.unlabeled_x.shape[0])

		data_pool = self.handler(self.unlabeled_x)
		for i in range(self.unlabeled_x.shape[0]):
			if i%5 == 0:
				print('adv {}/{}'.format(i, self.unlabeled_x.shape[0]))
			# print('Data_Pool ', data_pool[i])
			x, idx = data_pool[i]
			x = torch.from_numpy(x)
			dis[i] = self.cal_dis(x)

		self.model.to(self.device)
		# self.clf.cuda()
		idxs = dis.argsort()[:budget]
		return idxs


