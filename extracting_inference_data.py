import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, torch, torchvision, utils
from statistics import mode, multimode

def compute_ensemble_conf(prob_vectors, acc_branches, nr_branch_edge, target, device):

	nr_classes = len(prob_vectors[0])

	ensemble_prob_vector = torch.zeros(prob_vectors[0].shape, device=device)

	for i in range(nr_branch_edge):

		ensemble_prob_vector += acc_branches[i]*prob_vectors[i]

	ensemble_conf, ensemble_infered_class = torch.max(ensemble_prob_vector, 1)

	if(args.n_branches==3):
		ensemble_infered_class -= 1	
	
	correct = ensemble_infered_class.eq(target.view_as(ensemble_infered_class)).sum().item()
	return ensemble_conf.item(), ensemble_infered_class.item(), correct

def extract_ensemble_data(prob_vectors, acc_branches, n_exits, target, device):

	ensemble_conf_branch_list, infered_class_branch_list, correct_branch_list = [], [], []

	for nr_branch_edge in range(1, n_exits+1):

		ensemble_conf_branch, infered_class_branch, correct_branch = compute_ensemble_conf(prob_vectors, acc_branches, nr_branch_edge, target, device)

		ensemble_conf_branch_list.append(ensemble_conf_branch), infered_class_branch_list.append(infered_class_branch)
		correct_branch_list.append(correct_branch)

	return ensemble_conf_branch_list, infered_class_branch_list, correct_branch_list

def extract_naive_ensemble(conf_branches, infered_class_branches, n_exits, target, device):

	conf_branches = [conf.item() for conf in conf_branches]

	infered_class_branches_list = [infered_class.item() for infered_class in infered_class_branches]

	conf_list, infered_class_list, correct_list = [], [], []


	for i in range(n_exits):

		conf_edge = conf_branches[:(i+1)]

		max_idx = np.argmax(conf_edge)

		mode_list = multimode(infered_class_branches_list)

		correct = int(target.item() in mode_list)

		conf_list.append(conf_edge[max_idx]), correct_list.append(correct)

	return conf_list, correct_list



def run_ensemble_inference_data(model, test_loader, acc_branches, n_branches, distortion_type_model, distortion_type_data, distortion_lvl, device):

	n_exits = n_branches + 1
	ensemble_conf_list, ensemble_infered_class_list, ensemble_correct_list, target_list = [], [], [], []

	model.eval()

	with torch.no_grad():
		for (data, target) in tqdm(test_loader):

			data, target = data.to(device), target.to(device)

			prob_vectors, conf_branches, infered_class_branches, _ = model(data)

			ensemble_conf, ensemble_infered_class, ensemble_correct = extract_ensemble_data(prob_vectors, acc_branches, n_exits, target, device)
			
			ensemble_conf_list.append(ensemble_conf), ensemble_infered_class_list.append(ensemble_infered_class)
			ensemble_correct_list.append(ensemble_correct)
			target_list.append(target.item())


			del data, target
			torch.cuda.empty_cache()


	ensemble_conf_list = np.array(ensemble_conf_list)
	ensemble_infered_class_list = np.array(ensemble_infered_class_list)
	ensemble_correct_list = np.array(ensemble_correct_list)

	ensemble_results = {"distortion_type_model": [distortion_type_model]*len(target_list),
	"distortion_type_data": [distortion_type_data]*len(target_list), "distortion_lvl": [distortion_lvl]*len(target_list), 
	"target": target_list}

	for i in range(n_exits):
		ensemble_results.update({"ensemble_conf_branch_%s"%(i+1): ensemble_conf_list[:, i], 
			"ensemble_infered_class_branches_%s"%(i+1): ensemble_infered_class_list[:, i], 
			"ensemble_correct_branch_%s"%(i+1): ensemble_correct_list[:, i]})

	return ensemble_results


def run_inference_data(model, test_loader, n_branches, distortion_type_model, distortion_type_data, distortion_lvl, device):

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list = [], []
	naive_ensemble_conf_list, naive_ensemble_correct_list = [], []
	inference_time_branches_list = []

	model.eval()

	with torch.no_grad():
		for (data, target) in tqdm(test_loader):

			data, target = data.to(device), target.to(device)

			prob_vectors, conf_branches, infered_class_branches, inference_time_branches = model(data)

			#ensemble_conf, ensemble_infered_class, ensemble_correct = extract_ensemble_data(prob_vectors, n_exits, target, device)

			naive_ensemble_conf, naive_ensemble_correct = extract_naive_ensemble(conf_branches, infered_class_branches, n_exits, 
				target, device)

			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			target_list.append(target.item())
			
			naive_ensemble_conf_list.append(naive_ensemble_conf), naive_ensemble_correct_list.append(naive_ensemble_correct)

			inference_time_branches_list.append(inference_time_branches)


			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)

	naive_ensemble_conf_list = np.array(naive_ensemble_conf_list)
	naive_ensemble_correct_list = np.array(naive_ensemble_correct_list)

	inference_time_branches_list = np.array(inference_time_branches_list)

	results = {"distortion_type_model": [distortion_type_model]*len(target_list),
	"distortion_type_data": [distortion_type_data]*len(target_list), "distortion_lvl": [distortion_lvl]*len(target_list), 
	"target": target_list}

	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i],
			"naive_ensemble_conf_branch_%s"%(i+1): naive_ensemble_conf_list[:, i],
			"naive_ensemble_correct_branch_%s"%(i+1): naive_ensemble_correct_list[:, i],
			"inference_time_branches_%s"%(i+1): inference_time_branches_list[:, i]}) 

	return results

def save_result(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def compute_acc_branches(result, n_branches):

	acc_list = []
	n_exits = n_branches+1

	for i in range(n_exits):

		acc_branch = sum(result["correct_branch_%s"%(i+1)])/len(result["correct_branch_%s"%(i+1)])
		acc_list.append(acc_branch)

	return acc_list

def extracting_inference_data(model, input_dim, dim, inference_data_path, dataset_path, indices_path, device, 
	distortion_type_model, distortion_type_data):

	distortion_lvl_list = config.distortion_level_dict[distortion_type_data]

	for distortion_lvl in distortion_lvl_list:
		print("Distortion Level: %s"%(distortion_lvl))

		_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path, input_dim, dim, distortion_type_data, distortion_lvl)

		result = run_inference_data(model, test_loader, args.n_branches, distortion_type_model, distortion_type_data, distortion_lvl, device)
		acc_branches = compute_acc_branches(result, args.n_branches)

		ensemble_results = run_ensemble_inference_data(model, test_loader, acc_branches, args.n_branches, distortion_type_model, distortion_type_data, 
			distortion_lvl, device)

		result.update(ensemble_results)

		save_result(result, inference_data_path)


def main(args):

	distorted_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"%s_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.distortion_type_model, args.n_branches, args.model_id) )

	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = config.nr_class_dict[args.dataset_name]
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]

	#Load the trained early-exit DNN model.
	ee_model = utils.load_ee_dnn(args, distorted_model_path, n_classes, dim, device)
	ee_model.eval()

	dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

	for _ in range(10):
		_ = model(dummy_input)	
		
	extracting_inference_data(ee_model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
		device, args.distortion_type_model, distortion_type_data="pristine")

	extracting_inference_data(ee_model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
		device, args.distortion_type_model, distortion_type_data="gaussian_blur")

	extracting_inference_data(ee_model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
		device, args.distortion_type_model, distortion_type_data="gaussian_noise")


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

	args = parser.parse_args()
	main(args)
