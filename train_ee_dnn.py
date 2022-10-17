import os, time, sys, json, os, argparse, torch
import numpy as np
import utils, config, ee_nn


def main(args):

	dataset_path = os.path.join(config.DIR_NAME, "datasets", config.dataset_name)
	indices_path = os.path.join(config.DIR_NAME, "indices")

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	if not (os.path.exists(indices_path)):
		os.makedirs(indices_path)

	train_loader, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path)

	n_classes = config.nr_class_dict[args.dataset_name]

	#Instantiate the Early-exit DNN model.
	ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.dim, device, args.exit_type)
	#Load the trained early-exit DNN model.
	ee_model = ee_model.to(device)



if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "resnet18", "resnet152", "vgg16"], 
		help='DNN model name (default: %s)'%(config.model_name))

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--h_flip_prob', type=float, default=config.h_flip_prob, 
		help='Probability of Flipping horizontally.')

	parser.add_argument('--equalize_prob', type=float, default=config.equalize_prob, help='Probability of Equalize.')

	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)

	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda? Default: %s'%(config.cuda))

	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	args = parser.parse_args()

	main(args)