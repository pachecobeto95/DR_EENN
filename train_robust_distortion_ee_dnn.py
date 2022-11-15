import os, time, sys, json, os, argparse, torch, logging
import numpy as np
import pandas as pd
import utils, config, ee_nn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm



def main(args):

	dataset_path = config.dataset_path_dict[args.dataset_name]		
	indices_path = config.indices_path_dict[args.dataset_name]


	model_path = os.path.join(config.DIR_NAME, "undistorted_models", config.dataset_name, config.model_name, 
		"ee_model_%s_%s.pth"%(config.model_name, args.model_id))
	
	save_model_path = os.path.join(config.DIR_NAME, "distorted_models", config.dataset_name, config.model_name, 
		"%s_ee_model_%s_%s.pth"%(args.distortion_type, config.model_name, args.model_id))

	history_path = os.path.join(config.DIR_NAME, "history", config.dataset_name, config.model_name, 
		"distorted_%s_history_ee_model_%s_%s.csv"%(, config.model_name, args.model_id))

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	if not (os.path.exists(indices_path)):
		os.makedirs(indices_path)

	distortion_values = config.distortion_level_dict[args.distortion_type]
	train_loader, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path, distortion_values)

	n_classes = config.nr_class_dict[args.dataset_name]

	#Load the trained Early-exit DNN model.
	ee_model = utils.load_ee_dnn(args, model_path, n_classes, device)

	lr = [1.5e-4, 0.01]
	weight_decay = 0.0005
	n_exits = args.n_branches + 1

	loss_weights_dict = {"equal": np.ones(n_exits), "crescent": np.linspace(0.3, 1, n_exits), "decrescent": np.linspace(1, 0.3, n_exits)}

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam([{'params': ee_model.stages.parameters(), 'lr': lr[0]}, 
		{'params': ee_model.exits.parameters(), 'lr': lr[1]},
		{'params': ee_model.classifier.parameters(), 'lr': lr[0]}], weight_decay=weight_decay)

	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1, verbose=True)

	loss_weights = loss_weights_dict[args.loss_weights_type]

	epoch, count = 0, 0
	best_val_loss = np.inf
	df = pd.DataFrame()

	while (count < args.max_patience):
		epoch += 1
		current_result = {}
		train_result = utils.trainEEDNNs(ee_model, train_loader, optimizer, criterion, n_exits, epoch, device, loss_weights)
		val_result = utils.evalEEDNNs(ee_model, val_loader, criterion, n_exits, epoch, device, loss_weights)
		scheduler.step()
		current_result.update(train_result), current_result.update(val_result)
		df = df.append(pd.Series(current_result), ignore_index=True)
		df.to_csv(history_path)

		if (val_result["val_loss"] < best_val_loss):
			save_dict  = {}	
			best_val_loss = val_result["val_loss"]
			count = 0

			save_dict.update(current_result)
			save_dict.update({"model_state_dict": ee_model.state_dict(), "opt_state_dict": optimizer.state_dict()})
			torch.save(save_dict, model_save_path)

		else:
			count += 1
			print("Current Patience: %s"%(count))

	print("Stop! Patience is finished")




if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN: MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)

	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda? Default: %s'%(config.cuda))

	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: %s'%(config.distribution))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	parser.add_argument('--epochs', type=int, default=config.epochs, help='Epochs.')

	parser.add_argument('--max_patience', type=int, default=config.max_patience, help='Epochs.')

	parser.add_argument('--model_id', type=int, default=config.model_id, help='Epochs.')

	parser.add_argument('--distortion_type', type=str, help='Distortion Type.')

	parser.add_argument('--loss_weights_type', type=str, help='Loss Weights Type.')

	args = parser.parse_args()

	main(args)
