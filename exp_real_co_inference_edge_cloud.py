import os, config, time, utils
import requests, sys, json, os
import numpy as np
#from PIL import Image
#import pandas as pd
import argparse
#from utils import LoadDataset
#from requests.exceptions import HTTPError, ConnectTimeout
#from glob import glob
#from load_dataset import load_test_caltech_256
#from torchvision.utils import save_image
import logging, torch

def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	p_tar_list = [0.8, 0.82, 0.83, 0.85, 0.9]

	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	logPath = "./logTest_%s_%s.log"%(args.model_name, args.dataset_name)

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	
	#This line defines the number of side branches processed at the edge
	nr_branch_edge = np.arange(2, args.n_branches+1)

	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

	n_classes = config.nr_class_dict[args.dataset_name]
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]


	#print("Sending Confs")
	#logging.debug("Sending Confs")

	#sendModelConf(config.urlConfModelEdge, config.nr_branch_model, args.dataset_name, args.model_name, args.location)
	#sendModelConf(config.urlConfModelCloud, config.nr_branch_model, args.dataset_name, args.model_name, args.location)
	
	#print("Finish Confs")
	#logging.debug("Finish Confs")

	distortion_lvl_list = config.distortion_level_dict["gaussian_blur"]

	for distortion_lvl in distortion_lvl_list:
		print("Distortion Level: %s"%(distortion_lvl))

		_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path, input_dim, dim, "gaussian_blur", distortion_lvl)

		for i, (data, target) in enumerate(test_loader, 1):

			data, target = data.to(device), target.to(device)

			sys.exit()



if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")

	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], 
		help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default=config.model_name, 
		choices=["mobilenet"], help='DNN model name (default: MobileNet)')

	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')

	#parser.add_argument('--location', type=str, choices=["ohio", "sp"], help='Location of Cloud Server')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--distortion_prob', type=float, default=1)

	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')

	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, help='Size of train batch.')


	args = parser.parse_args()


	main(args)
