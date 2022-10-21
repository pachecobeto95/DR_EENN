import os, time, sys, json, os, argparse, torch, logging
import numpy as np
import pandas as pd
import utils, config, ee_nn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from ee_calibration import calibrating_early_exit_dnn, run_early_exit_inference


def save_result(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def save_results(no_calib_result, global_calib_result, per_branch_calib_result, all_samples_calib_result, resultsDict):
	save_result(no_calib_result, resultsDict["no_calib"])
	save_result(global_calib_result, resultsDict["global_calib"])
	save_result(per_branch_calib_result, resultsDict["per_branch_calib"])
	save_result(all_samples_calib_result, resultsDict["all_samples_calib"])

def run_calibration(model, test_loader, val_loader, p_tar_list, n_branches, device, resultsDict, temperatureDict):
	for p_tar in p_tar_list:
		print("P_tar: %s"%(p_tar))

		calib_models_dict = calibrating_early_exit_dnn(model, val_loader, p_tar, n_branches, device, temperatureDict)

		no_calib_result = run_early_exit_inference(model, test_loader, p_tar, n_branches, device, model_type="no_calib")

		global_calib_result = run_early_exit_inference(calib_models_dict["global_calib"], test_loader, p_tar, n_branches, device, 
			model_type="global_calib")

		per_branch_calib_result = run_early_exit_inference(calib_models_dict["per_branch_calib"], test_loader, p_tar, n_branches, device, 
			model_type="per_branch_calib")

		all_samples_calib_result = run_early_exit_inference(calib_models_dict["all_samples_calib"], test_loader, p_tar, 
			n_branches, device, model_type="all_samples_calib")

		save_results(no_calib_result, global_calib_result, per_branch_calib_result, all_samples_calib_result, resultsDict)


def main(args):
	dataset_path = os.path.join(config.DIR_NAME, "datasets", config.dataset_name, "256_ObjectCategories")
	indices_path = os.path.join(config.DIR_NAME, "indices")
	model_save_path = os.path.join(config.DIR_NAME, "models", config.dataset_name, config.model_name, 
		"%s_ee_model_%s_%s.pth"%(args.distortion_type, config.model_name, args.model_id))
	
	result_path = os.path.join(config.DIR_NAME, "result_calib", config.dataset_name, config.model_name)
	
	no_calib_result_path =  os.path.join(result_path, "no_calib_results_%s.csv"%(args.model_id))
	save_global_calib_path =  os.path.join(result_path, "global_calib_results_%s.csv"%(args.model_id))
	save_per_branch_calib_path =  os.path.join(result_path, "per_branch_calib_results_%s.csv"%(args.model_id))
	save_all_samples_calib_path =  os.path.join(result_path, "all_samples_calib_results_%s.csv"%(args.model_id))

	resultsDict = {"no_calib": no_calib_result_path, 
	"global_calib": save_global_calib_path, 
	"per_branch_calib": save_per_branch_calib_path,
	"all_samples_calib": save_all_samples_calib_path}

	save_global_temp_path =  os.path.join(result_path, "global_calib_results_%s.csv"%(args.model_id))
	save_per_branch_temp_path =  os.path.join(result_path, "per_branch_calib_results_%s.csv"%(args.model_id))
	save_all_samples_temp_path =  os.path.join(result_path, "all_samples_calib_results_%s.csv"%(args.model_id))

	temperatureDict = {"global_calib": save_global_calib_path, 
	"per_branch_calib": save_per_branch_calib_path,
	"all_samples_calib": save_all_samples_calib_path}

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	distortion_values = config.distortion_level_dict[args.distortion_type]
	train_loader, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path, distortion_values)

	n_classes = config.nr_class_dict[args.dataset_name]

	#Instantiate the Early-exit DNN model.
	ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.dim, device, args.exit_type)
	#Load the trained early-exit DNN model.
	ee_model = ee_model.to(device)
	ee_model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])


	p_tar_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	run_calibration(ee_model, test_loader, val_loader, p_tar_list, args.n_branches, device, resultsDict, temperatureDict)




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

	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)

	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda? Default: %s'%(config.cuda))

	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')

	parser.add_argument('--epochs', type=int, default=config.epochs, help='Epochs.')

	parser.add_argument('--max_patience', type=int, default=config.max_patience, help='Epochs.')

	parser.add_argument('--model_id', type=int, default=1, help='Epochs.')

	args = parser.parse_args()

	main(args)
