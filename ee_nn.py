import torchvision, torch, os, sys, time, math
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pthflops import count_ops
from torch import Tensor

"""
(18): ConvBNActivation(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1000, bias=True)
"""

class EarlyExitBlock(nn.Module):
	def __init__(self, input_shape, last_channel, n_classes, exit_type, device):
		super(EarlyExitBlock, self).__init__()
		self.input_shape = input_shape
		_, channel, width, height = input_shape
		
		self.layers = nn.ModuleList()

		if (exit_type == 'bnpool'):
			self.layers.append(nn.BatchNorm2d(channel))
			self.layers.append(nn.AdaptiveAvgPool2d(1))

		elif(exit_type == 'conv'):
			self.layers.append(nn.Conv2d(channel, last_channel, kernel_size=(1, 1)))
			self.layers.append(nn.BatchNorm2d(last_channel))
			self.layers.append(nn.ReLU6(inplace=True))

		elif(exit_type == 'pooling'):
			self.layers.append(nn.BatchNorm2d(channel))
			self.layers.append(nn.MaxPool2d(kernel_size))

		elif(exit_type == 'plain'):
			self.layers = nn.ModuleList()

		#This line defines the data shape that fully-connected layer receives.
		current_channel, current_width, current_height = self.get_current_data_shape()

	def get_current_data_shape(self):
		print("Before")
		print(self.input_shape)
		_, channel, width, height = self.input_shape
		temp_layers = nn.Sequential(*self.layers)

		input_tensor = torch.rand(1, channel, width, height)
		_, output_channel, output_width, output_height = temp_layers(input_tensor).shape
		print("After")
		print(output_channel, output_width, output_height)		
		return output_channel, output_width, output_height

	def forward(self, x):
		return x


class Early_Exit_DNN(nn.Module):
	def __init__(self, model_name: str, n_classes: int, pretrained: bool, n_branches: int, input_dim: int, 
		device, exit_type: str, distribution="predefined", ee_point_location=10):

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
		self.ee_point_location = ee_point_location

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

	def is_suitable_for_exit(self, nr_block):

		if(self.distribution=="predefined"):
			return (self.stage_id < self.n_branches) and (nr_block > self.ee_point_location)

		else:
			intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers))).to(self.device)
			x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
			current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
			return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

	def add_exit_block(self):
		"""
		This method adds an early exit in the suitable position.
		"""
		input_tensor = torch.rand(1, 3, self.input_dim, self.input_dim)

		self.stages.append(nn.Sequential(*self.layers))
		x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)

		feature_shape = nn.Sequential(*self.stages)(x).shape
		
		self.exits.append(EarlyExitBlock(feature_shape, self.last_channel, self.n_classes, self.exit_type, self.device).to(self.device))
		self.layers = nn.ModuleList()
		self.stage_id += 1    

	def early_exit_mobilenet(self):

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.stage_id = 0

		self.last_channel = 1280

		# Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
		backbone_model = models.mobilenet_v2(self.pretrained).to(self.device)

		# This obtains the flops total of the backbone model
		self.total_flops = self.countFlops(backbone_model)

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		#self.threshold_flop_list = self.where_insert_early_exits()

		for nr_block, block in enumerate(backbone_model.features.children()):
			
			self.layers.append(block)
			if (self.is_suitable_for_exit(nr_block)):
				self.add_exit_block()

		self.layers.append(nn.AdaptiveAvgPool2d(1))
		self.stages.append(nn.Sequential(*self.layers))

		self.classifier = backbone_model.classifier
		self.softmax = nn.Softmax(dim=1)
