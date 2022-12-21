import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, torch, torchvision, utils
import torchvision.models as models
import torch.nn as nn



def run_inference(args, model, input_data, device):

	model.eval()
	softmax = nn.Softmax(dim=1)

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	inference_time_list = []

	with torch.no_grad():
		for i in tqdm(range(args.n_rounds)):

			starter.record()

			_ = model(input_data)

			ender.record()
			torch.cuda.synchronize()
			curr_time = starter.elapsed_time(ender)

			inference_time_list.append(curr_time)

			torch.cuda.empty_cache()

	results = {"inference_time": inference_time_list}

	return results


def save_result(result, save_path):
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def extracting_inference_time_backbone(args, model, input_data, input_dim, dim, save_path, device):

	result = run_inference_data(args, model, input_data, device)

	save_result(result, save_path)


def main(args):

	model_save_path = os.path.join(config.DIR_NAME, "models", config.dataset_name, config.model_name, 
		"%s_backbone_model_%s_%s.pth"%(args.distortion_type, config.model_name, args.model_id))


	inference_time_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_time_backbone_id_%s_final_final.csv"%(args.model_id))

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = 257

	#Load the trained backbone DNN model.
	model = models.mobilenet_v2(pretrained=True)
	model.classifier[1] = nn.Linear(1280, n_classes)

	model.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
	model = model.to(device)

	dummy_input = torch.randn(1, 3, 224,224, dtype=torch.float).to(device)

	for _ in range(10):
		_ = model(dummy_input)	


	extracting_inference_time_backbone(args, model, dummy_input, args.input_dim, args.dim, inference_time_path, device)


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
	parser.add_argument('--distortion_type', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)
	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))


	args = parser.parse_args()
	main(args)
