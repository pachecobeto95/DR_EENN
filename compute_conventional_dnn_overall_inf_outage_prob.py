import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, math
import scipy.stats as st


def main(args):

	overall_inf_outage_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet", 
		"overall_inf_outage_prob_%s_branches_id_%s.csv"%(args.n_branches, args.model_id))
	
	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet",  
		"inference_data_backbone_id_%s_final_final.csv"%(args.model_id))

	threshold_list = np.arange(config.threshold_start, config.threshold_end, config.threshold_step)

	df_inf_data = pd.read_csv(inference_data_path)

	print(df_inf_data.columns)


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='UCB using MobileNet')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--distortion_type_model', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda ?')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--n_rounds', type=float, default=config.n_rounds_outage)
	parser.add_argument('--n_batches', type=int, default=config.batch_size_outage)

	args = parser.parse_args()
	main(args)
