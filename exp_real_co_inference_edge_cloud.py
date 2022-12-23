import os, config, time, utils
import requests, sys, json, os
import numpy as np
#from PIL import Image
#import pandas as pd
import argparse
#from utils import LoadDataset
from requests.exceptions import HTTPError, ConnectTimeout
#from glob import glob
#from load_dataset import load_test_caltech_256
#from torchvision.utils import save_image
import logging, torch
from PIL import Image



def sendImage(url, filePath, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl):
	
	data_dict = {"p_tar": str(p_tar), "target": str(target), "nr_branch_edge": str(nr_branch_edge), "distortion_type": distortion_type,
	"distortion_lvl": str(distortion_lvl)}

	files = [
	('img', (filePath, open(filePath, 'rb'), 'application/octet')),
	('data', ('data', json.dumps(data_dict), 'application/json')),]

	try:
		r = requests.post(url, files=files, timeout=config.timeout)
		r.raise_for_status()
	
	except HTTPError as http_err:
		raise SystemExit(http_err)

	except requests.Timeout:
		logging.warning("Timeout")
		pass


		

def sendDistortedImage(imgPath, target, nr_branch_edge, p_tar, distortion_lvl, distortion_type):
	sendImage(config.url_ee, imgPath, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl)
	sendImage(config.url_ensemble, imgPath, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl)
	sys.exit()	
	#sendImage(config.url_backbone, imgPath, target, nr_branch_edge, p_tar, distortion_lvl)


def sendDistortedImageSet(dataset_path_list, target_list, distortion_lvl_list, p_tar, nr_branch_edge):

	for imgPath, target, distortion_lvl in zip(dataset_path_list, target_list, distortion_lvl_list):
		sendDistortedImage(imgPath, target, nr_branch_edge, p_tar, distortion_lvl, distortion_type="gaussian_blur")


def inferenceTimeExp(distorted_datasetPath, p_tar_list, nr_branch_edge_list):


	file_path_list, distortion_lvl_list, target_list = utils.getImageFilePath(distorted_datasetPath)	


	for p_tar in p_tar_list:

		for nr_branch_edge in nr_branch_edge_list:

			print("p_tar: %s, Number Branches at Edge: %s"%(p_tar, nr_branch_edge) )

			sendDistortedImageSet(file_path_list, target_list, distortion_lvl_list, p_tar, nr_branch_edge)


def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	distorted_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"pristine_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.n_branches, args.model_id) )


	p_tar_list = [0.8, 0.82, 0.83, 0.85, 0.9]

	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	logPath = "./logTest_%s_%s.log"%(args.model_name, args.dataset_name)

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	
	#This line defines the number of side branches processed at the edge
	nr_branch_edge_list = np.arange(2, args.n_branches+1)

	#device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
	inferenceTimeExp(config.distorted_dataset_path, p_tar_list, nr_branch_edge_list)



if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")

	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], 
		help='Dataset name (default: Caltech-256)')

	parser.add_argument('--model_name', type=str, default=config.model_name, 
		choices=["mobilenet"], help='DNN model name (default: MobileNet)')

	parser.add_argument('--n_branches', type=int, default=3, help='Number of exit exits.')

	#parser.add_argument('--location', type=str, choices=["ohio", "sp"], help='Location of Cloud Server')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--distortion_prob', type=float, default=1)

	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')

	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, help='Size of train batch.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, help='Exit type.')
	parser.add_argument('--distribution', type=str, default=config.distribution, help='Distribution of early exits.')
	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Pretrained ?')

	parser.add_argument('--cuda', type=bool, default=True, help='Cuda ?')

	args = parser.parse_args()


	main(args)
