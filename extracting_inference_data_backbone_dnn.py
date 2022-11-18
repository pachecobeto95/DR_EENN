import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, torch, torchvision, utils



def run_inference_data(model, test_loader, distortion_type_data, distortion_lvl, device):

	conf_list, infered_class_list, target_list, correct_list = [], [], [], []

	model.eval()

	with torch.no_grad():
		for (data, target) in tqdm(test_loader):

			data, target = data.to(device), target.to(device)

			output = model(data)

			conf_list.append(conf.item()), infered_class_list.append(inf_class.item())    
			correct_list.append(inf_class.eq(target.view_as(inf_class)).sum().item())			
			target_list.append(target.item())

			del data, target, conf, inf_class, output
			torch.cuda.empty_cache()

	print("Acc: %s"%(sum(correct_list)/len(correct_list)))

	results = {"distortion_type_data": [distortion_type_data]*len(target_list), 
	"distortion_lvl": [distortion_lvl]*len(target_list), 
	"target": target_list, "conf": conf_list, "infered_class": infered_class_list,
	"correct": correct_list}

	return results

def save_result(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def extracting_inference_data(model, input_dim, dim, inference_data_path, dataset_path, indices_path, device, 
	distortion_type_data):

	distortion_lvl_list = config.distortion_level_dict[distortion_type_data]

	for distortion_lvl in distortion_lvl_list:
		print("Distortion Level: %s"%(distortion_lvl))

		_, _, test_loader = utils.load_caltech256(args, dataset_path, indices_path, input_dim, dim, distortion_type_data, distortion_lvl)

		result = run_inference_data(model, test_loader, distortion_type_data, distortion_lvl, device)

		save_result(result, inference_data_path)


def main(args):

	model_save_path = os.path.join(config.DIR_NAME, "models", config.dataset_name, config.model_name, 
		"%s_backbone_model_%s_%s.pth"%(args.distortion_type, config.model_name, args.model_id))


	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_backbone_id_%s.csv"%(args.model_id))

	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = 257
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]

	#Load the trained backbone DNN model.
	model = models.mobilenet_v2(pretrained=True)
	model.classifier[1] = nn.Linear(1280, n_classes)

	model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])

	extracting_inference_data(model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
		device, distortion_type_data="pristine")

	extracting_inference_data(model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
		device, distortion_type_data="gaussian_blur")

	extracting_inference_data(model, input_dim, dim, inference_data_path, dataset_path, indices_path, 
		device, distortion_type_data="gaussian_noise")


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='UCB using MobileNet')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda ?')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, help='Size of train batch.')
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')
	parser.add_argument('--distortion_prob', type=float, default=1)

	args = parser.parse_args()
	main(args)
