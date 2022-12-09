import numpy as np
import pandas as pd
import argparse, config, os, sys
import matplotlib.pyplot as plt


def plotAccuracyDistortionLevel(df, threshold, n_branches_edge, distortion_levels, plot_dict, distortion_type):

	df = df[(df.n_branches_edge == n_branches_edge) & (df.threshold == threshold) & (df.distortion_type_data==distortion_type)]

	print(df.shape)
	print(len(distortion_levels))

	plt.plot(, )

def main(args):

	results_path = os.path.join(config.DIR_NAME, "ensemble_analysis_results", 
		"pristine_model_ensemble_analysis_%s_branches_%s_%s.csv"%(args.n_branches, args.model_name, args.dataset_name))

	df = pd.read_csv(results_path)
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

	blur_levels = [0] + config.distortion_level_dict["gaussian_blur"]
	noise_levels = [0] + config.distortion_level_dict["gaussian_noise"]

	for threshold in config.threshold_list:
		for n_branch_edge in range(1, args.n_branches+1):

			plotAccuracyDistortionLevel(df, threshold, n_branch_edge, blur_levels, config.plot_dict, distortion_type="gaussian_blur")

			plotAccuracyDistortionLevel(df, threshold, n_branch_edge, noise_levels, config.plot_dict, distortion_type="gaussian_noise")

if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Plot results of Ensemble in Early-exit DNNs.')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--distortion_type_model', type=str, default=config.distortion_type, help='Distortion Type of the model.')
	parser.add_argument('--n_branches', type=int, default=config.n_branches, help='Number of exit exits.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda ?')

	args = parser.parse_args()
	main(args)
