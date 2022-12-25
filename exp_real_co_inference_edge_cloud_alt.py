import os, config, time, utils, requests, sys, json, os, argparse, logging, torch
import numpy as np
from requests.exceptions import HTTPError, ConnectTimeout
from PIL import Image
import torchvision.transforms as transforms



def sendImage(url, img, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl):
	
	data_dict = {"img": feature_map.detach().cpu().numpy().tolist(), "p_tar": str(p_tar), "target": str(target), "nr_branch_edge": str(nr_branch_edge), 
	"distortion_type": distortion_type,
	"distortion_lvl": str(distortion_lvl), "mode": "_alt"}

	try:
		r = requests.post(url, json=data, timeout=config.timeout)
		r.raise_for_status()
	
	except HTTPError as http_err:
		raise SystemExit(http_err)

	except requests.Timeout:
		logging.warning("Timeout")
		pass


def applyDistortiontransformation(data, distortion_lvl):

	data = data[0]

	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)

	transformation = transforms.Compose([transforms.ToPILImage(),
		transforms.RandomApply([utils.DistortionApplier("gaussian_blur", distortion_lvl)], p=1),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	return transformation(data)


def sendDistortedImage(data, target, nr_branch_edge, p_tar, distortion_lvl, distortion_type):
	sendImage(config.url_ee_alt, data, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl)
	sendImage(config.url_ensemble_alt, data, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl)
	sendImage(config.url_naive_ensemble_alt, data, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl)
	sendImage(config.url_cloud_backbone_alt, data, target, nr_branch_edge, p_tar, distortion_type, distortion_lvl)

def sendDistortedImageSet(dataset_path, indices_path, distortion_lvl_list, p_tar, nr_branch_edge, n_classes, input_dim, dim):

	print("entrou")

	test_loader = utils.load_caltech256_inference_time_exp(args, dataset_path, indices_path, input_dim, dim)
	
	print("saiu")	

	for i, (data, target) in enumerate(test_loader, 1):

		print("Image: %s"%(i))

		target = target.item()

		for distortion_lvl in distortion_lvl_list:

			data = applyDistortiontransformation(data, distortion_lvl)

			sendDistortedImage(data, target, nr_branch_edge, p_tar, distortion_lvl, distortion_type="gaussian_blur")



def inferenceTimeExp(dataset_path, indices_path, distortion_lvl_list, n_classes, input_dim, dim, p_tar_list, nr_branch_edge_list):

	for p_tar in p_tar_list:

		for nr_branch_edge in nr_branch_edge_list:

			#logging.debug("p_tar: %s, Number Branches at Edge: %s"%(p_tar, nr_branch_edge))
			print("p_tar: %s, Number Branches at Edge: %s"%(p_tar, nr_branch_edge))

			sendDistortedImageSet(dataset_path, indices_path, distortion_lvl_list, p_tar, nr_branch_edge, n_classes, input_dim, dim)


def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	distorted_model_path =  os.path.join(config.DIR_NAME, "models", args.dataset_name, args.model_name, 
		"pristine_ee_model_mobilenet_%s_branches_id_%s.pth"%(args.n_branches, args.model_id) )


	p_tar_list = [0.8, 0.82, 0.83, 0.85, 0.9]

	indices_path = os.path.join(config.DIR_NAME, "indices")
	
	dataset_path = config.dataset_path_dict[args.dataset_name]

	logPath = "./log_inference_time_exp_%s_%s_alt.log"%(args.model_name, args.dataset_name)

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
	
	#This line defines the number of side branches processed at the edge
	nr_branch_edge_list = np.arange(2, args.n_branches+1)

	distortion_lvl_list = config.distortion_level_dict["gaussian_blur"]

	device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')

	n_classes = config.nr_class_dict[args.dataset_name]
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]


	#device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
	inferenceTimeExp(dataset_path, indices_path, distortion_lvl_list, n_classes, input_dim, dim, p_tar_list, nr_branch_edge_list)



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
