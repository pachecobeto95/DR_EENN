import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, torch, torchvision, utils
#from statistics import mode, multimode


def run_inference(args, model, input_data, n_branches, device):

	n_exits = n_branches + 1
	inference_time_branches_list = []

	results = {}

	model.eval()

	with torch.no_grad():
		for i in tqdm(range(args.n_rounds)):

			_, _, _, inference_time_branches = model(input_data)

			inference_time_branches_list.append(inference_time_branches)

			torch.cuda.empty_cache()

	inference_time_branches_list = np.array(inference_time_branches_list)

	for i in range(n_exits):
		results.update({"inference_time_branches_%s"%(i+1): inference_time_branches_list[:, i]}) 

	return results


def save_result(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def extracting_inference_time(args, model, input_data, input_dim, dim, save_path, device):

	result = run_inference(args, model, input_data, args.n_branches, device)

	save_result(result, save_path)


def main(args):

	distorted_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"%s_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.distortion_type_model, args.n_branches, args.model_id) )

	inference_time_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_time_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = config.nr_class_dict[args.dataset_name]
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]

	#Load the trained early-exit DNN model.
	ee_model = utils.load_ee_dnn(args, distorted_model_path, n_classes, dim, device)
	ee_model.eval()

	dummy_input = torch.randn(1, 3, 224,224, dtype=torch.float).to(device)

	for _ in range(10):
		_ = ee_model(dummy_input)	


	extracting_inference_time(args, ee_model, dummy_input, input_dim, dim, inference_time_path, device)




if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='UCB using MobileNet')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--distortion_type_model', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda ?')
	parser.add_argument('--exit_type', type=str, default=config.exit_type, help='Exit type.')
	parser.add_argument('--distribution', type=str, default=config.distribution, help='Distribution of early exits.')
	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Pretrained ?')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--calib_type', type=str, default="no_calib", help='Calibration Type.')
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, help='Size of train batch.')
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')
	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)
	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))
	parser.add_argument('--distortion_prob', type=float, default=1)
	parser.add_argument('--n_rounds', type=float, default=10000)


	args = parser.parse_args()
	main(args)
