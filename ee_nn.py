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
		self.early_exit_point_location = 10

		build_early_exit_dnn = self.select_dnn_architecture_model()
		build_early_exit_dnn()

	def select_dnn_architecture_model(self):
		"""
		This method selects the backbone to insert the early exits.
		"""

		architecture_dnn_model_dict = {"mobilenet": self.early_exit_mobilenet}

		return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)

	def invalid_model(self):
		raise Exception("This DNN backbone model has not implemented yet.")


	def countFlops(self, model):
	    input_data = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
	    flops, all_data = count_ops(model, input_data, print_readable=False, verbose=False)
	    return flops

	def early_exit_mobilenet(self):

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.stage_id = 0

		last_channel = 1280

		# Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
		backbone_model = models.mobilenet_v2(self.pretrained).to(self.device)

		#print(backbone_model)
		#sys.exit()

		# This obtains the flops total of the backbone model
		self.total_flops = self.countFlops(backbone_model)

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		#self.threshold_flop_list = self.where_insert_early_exits()

		for nr_block, block in enumerate(backbone_model.features.children()):
			
			self.layers.append(block)

			if(nr_block > self.early_exit_point_location):
				self.add_exit_block()



		self.stages.append(nn.Sequential(*self.layers))

		self.classifier = backbone_model.classifier

		#self.set_device()
		self.softmax = nn.Softmax(dim=1)
