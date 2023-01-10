import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, torch, torchvision, utils
#from statistics import mode, multimode
from pthflops import count_ops


def load_distorted_ee_dnn(args, pristine_model_path, blur_model_path, noise_model_path, n_classes, dim, device):

	pristine_ee_model = utils.load_ee_dnn(args, pristine_model_path, n_classes, dim, device)
	blur_ee_model = utils.load_ee_dnn(args, blur_model_path, n_classes, dim, device)
	noise_ee_model = utils.load_ee_dnn(args, noise_model_path, n_classes, dim, device)

	pristine_ee_model.eval(), blur_ee_model.eval(), noise_ee_model.eval()

	return [pristine_ee_model, blur_ee_model, noise_ee_model]



def run_inference_data(model, test_loader, n_branches, dist_type_model, dist_type_data, distortion_lvl, device):

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list, correct_list, entropy_list = [], [], [], [], []
	#flop_list, inference_time_list = [], []

	model.eval()

	with torch.no_grad():
		for data, target in test_loader:

			data, target = data.to(device), target.to(device)

			entropy_branches, conf_branches, infered_class_branches = model(data)

			conf_branches_list.append([conf.item() for conf in conf_branches]), entropy_list.append(entropy_branches)
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			target_list.append(target.item())
			#inference_time_list.append(inference_time)
			#lop_list.append(flop)

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	entropy_list = np.array(entropy_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)
	target_list = np.array(target_list)
	#inference_time_list = np.array(inference_time_list)
	#flop_list = np.array(flop_list)

	results = {"distortion_type_model": [dist_type_model]*len(target_list),
	"distortion_type_data": [dist_type_data]*len(target_list), "distortion_lvl": [distortion_lvl]*len(target_list), 
	"target": target_list}
	#"inference_time": inference_time_list,
	#"flop": flop_list}

	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i],
			"entropy_branch_%s"%(i+1): entropy_list[:, i]}) 

	return results


def save_result(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def extracting_inference_data(args, model, input_dim, dim, inference_data_path, dataset_path, indices_path, device, dist_type_model, dist_level_dict, dist_type_data):

	print(dist_type_data, dist_level_dict[dist_type_data])

	distortion_lvl_list = dist_level_dict[dist_type_data]

	for distortion_lvl in distortion_lvl_list:
		print("Distortion Level: %s"%(distortion_lvl))

		_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path, input_dim, dim, dist_type_data, distortion_lvl)

		result = run_inference_data(model, test_loader, args.n_branches, dist_type_model, dist_type_data, distortion_lvl, device)

		save_result(result, inference_data_path)



def main(args):


	inference_data_path = os.path.join(config.DIR_NAME, "inference_distorted_data", args.dataset_name, args.model_name, 
		"inference_distorted_data_%s_branches_id_%s.csv"%(args.n_branches, args.model_id))

	blur_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"gaussian_blur_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.n_branches, args.model_id) )

	noise_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"gaussian_noise_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.n_branches, args.model_id) )

	pristine_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"pristine_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.n_branches, args.model_id) )


	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = config.nr_class_dict[args.dataset_name]
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]

	#Load the trained early-exit DNN model.
	distorted_ee_models = load_distorted_ee_dnn(args, pristine_model_path, blur_model_path, noise_model_path, n_classes, dim, device)

	distortion_type_list = ["pristine", "gaussian_blur", "gaussian_noise"]

	dist_level_dict = {"pristine": [0], "gaussian_blur": [1, 2, 3, 4, 5], 
	"gaussian_noise": [1, 10, 20, 30, 40]}


	for ee_model, distortion_type_model in zip(distorted_ee_models, distortion_type_list):

		print("Model: %s"%(distortion_type_model))

		extracting_inference_data(args, ee_model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
			device, distortion_type_model, dist_level_dict, dist_type_data="pristine")

		extracting_inference_data(args, ee_model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
			device, distortion_type_model, dist_level_dict, dist_type_data="gaussian_blur")

		extracting_inference_data(args, ee_model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
			device, distortion_type_model, dist_level_dict, dist_type_data="gaussian_noise")



if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Extracting Inference Distorted Data.')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	#parser.add_argument('--distortion_type_model', type=str, help='Model trained with distortion type.')
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

	args = parser.parse_args()
	main(args)
