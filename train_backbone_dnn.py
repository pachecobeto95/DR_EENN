import os, time, sys, json, os, argparse, torch, logging
import numpy as np
import pandas as pd
import utils, config, ee_nn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision.models as models


def main(args):


	dataset_path = os.path.join(config.DIR_NAME, "datasets", config.dataset_name, "257_ObjectCategories")
	#indices_path = os.path.join(config.DIR_NAME, "indices")
	indices_path = os.path.join(config.DIR_NAME, "indices")

	model_save_path = os.path.join(config.DIR_NAME, "models", config.dataset_name, config.model_name, 
		"%s_backbone_model_%s_%s.pth"%(args.distortion_type, config.model_name, args.model_id))
	
	history_path = os.path.join(config.DIR_NAME, "history", config.dataset_name, config.model_name, 
		"history_%s_backbone_model_%s_%s.csv"%(args.distortion_type, config.model_name, args.model_id))

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	if not (os.path.exists(indices_path)):
		os.makedirs(indices_path)

	distortion_values = config.distortion_level_dict[args.distortion_type]	
	train_loader, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path, args.input_dim, args.dim, 
		args.distortion_type, distortion_values)

	model = models.mobilenet_v2(pretrained=True).to(device)

	model.classifier[1] = nn.Linear(1280, 257)


	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1, verbose=True)

	epoch, count = 0, 0
	best_val_loss = np.inf
	df = pd.DataFrame()

	while (count < args.max_patience):
		epoch += 1
		current_result = {}
		train_result = utils.trainBackboneDNN(model, train_loader, optimizer, criterion, n_exits, epoch, device)
		val_result = utils.evalBackboneDNN(model, val_loader, criterion, n_exits, epoch, device)
		scheduler.step()
		current_result.update(train_result), current_result.update(val_result)
		df = df.append(pd.Series(current_result), ignore_index=True)
		df.to_csv(history_path)

		if (val_result["val_loss"] < best_val_loss):
			save_dict  = {}	
			best_val_loss = val_result["val_loss"]
			count = 0

			save_dict.update(current_result)
			save_dict.update({"model_state_dict": model.state_dict(), "opt_state_dict": optimizer.state_dict()})
			torch.save(save_dict, model_save_path)

		else:
			count += 1
			print("Current Patience: %s"%(count))

	print("Stop! Patience is finished")




if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Backbone DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	#parser.add_argument('--h_flip_prob', type=float, default=config.h_flip_prob, 
	#	help='Probability of Flipping horizontally.')

	#parser.add_argument('--equalize_prob', type=float, default=config.equalize_prob, help='Probability of Equalize.')

	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)

	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda? Default: %s'%(config.cuda))

	parser.add_argument('--epochs', type=int, default=config.epochs, help='Epochs.')

	parser.add_argument('--max_patience', type=int, default=config.max_patience, help='Epochs.')

	parser.add_argument('--model_id', type=int, default=config.model_id, help='Epochs.')

	parser.add_argument('--lr', type=float, default=0.01, help='Epochs.')

	parser.add_argument('--weight_decay', type=float, default=0.0005, help='Epochs.')

	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')

	parser.add_argument('--distortion_prob', type=float, default=1)


	args = parser.parse_args()

	main(args)