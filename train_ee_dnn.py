import os, time, sys, json, os, argparse, torch
import numpy as np
import utils

def main(args):

	DIR_NAME = os.path.dirname(__file__)
	dataset_path = os.path.join(DIR_NAME, "datasets", "caltech256")
	indices_path = os.path.join(DIR_NAME, "indices")

	if not (os.path.exists(indices_path)):
		os.makedirs(indices_path)

	train_loader, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path)


if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default="caltech256", 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, default="caltech256", choices=["mobilenet", "resnet18", "resnet152", "vgg16"], 
		help='DNN model name (default: mobilenet)')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=0.1, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=256, 
		help='Train Batch Size. Default: %s'%(256))

	parser.add_argument('--h_flip_prob', type=float, default=0.5, 
		help='Train Batch Size. Default: %s'%(0.5))

	parser.add_argument('--input_dim', type=int, default=256, 
		help='Input Dim. Default: %s'%(256))

	parser.add_argument('--dim', type=int, default=224, 
		help='Dim. Default: %s'%(224))

	parser.add_argument('--seed', type=int, default=42, 
		help='Seed. Default: %s'%(42))

	args = parser.parse_args()

	main(args)