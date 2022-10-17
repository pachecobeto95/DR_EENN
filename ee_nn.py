import torchvision, torch, os, sys, time, math
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pthflops import count_ops
from torch import Tensor


class Early_Exit_DNN(nn.Module):
	def __init__(self, model_name: str, n_classes: int, pretrained: bool, n_branches: int, input_dim: int, 
		device, exit_type: str, distribution="linear"):

		super(Early_Exit_DNN, self).__init__()

		"""
		This classes builds an early-exit DNNs architectures
		Args:

		model_name: model name 
		n_classes: number of classes in a classification problem, according to the dataset
		pretrained: 
		n_branches: number of branches (early exits) inserted into middle layers
		input_dim: dimension of the input image
		exit_type: type of the exits
		distribution: distribution method of the early exit blocks.
		device: indicates if the model will processed in the cpu or in gpu
		    
		Note: the term "backbone model" refers to a regular DNN model, considering no early exits.
		"""

		self.model_name = model_name
		self.n_classes = n_classes
		self.pretrained = pretrained
		self.n_branches = n_branches
		self.input_dim = input_dim
		self.exit_type = exit_type
		self.distribution = distribution
		self.device = device
