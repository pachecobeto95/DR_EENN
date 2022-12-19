import numpy as np
import pandas as pd
import argparse, config, os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plotAccuracyDistortionLevel(df, threshold, n_branches_edge, distortion_levels, plot_dict, savePath, distortion_type):


	fig, ax = plt.subplots()

	df = df[(df.n_branches_edge == n_branches_edge) & (df.threshold == threshold) & (df.distortion_type_data==distortion_type)]

	saveFile = os.path.join(savePath, "%s_edge_accuracy_%s_branches_threshold_%s"%(distortion_type, n_branches_edge, threshold))

	plt.plot(distortion_levels, df.acc_backbone.values, label="Backbone DNN", 
		marker=plot_dict["marker"][0], linestyle=plot_dict["line_style"][0], color=plot_dict["color"][0])


	plt.plot(distortion_levels, df.overall_acc_ee.values, label="EE-DNN", 
		marker=plot_dict["marker"][1], linestyle=plot_dict["line_style"][1], color=plot_dict["color"][1])

	plt.plot(distortion_levels, df.overall_acc_ensemble.values, label="Ensemble EE", 
		marker=plot_dict["marker"][2], linestyle=plot_dict["line_style"][2], color=plot_dict["color"][2])


	plt.xlabel(plot_dict["x_axis"][distortion_type], fontsize=plot_dict["fontsize"])
	plt.ylabel("Overall Accuracy", fontsize=plot_dict["fontsize"])
	ax.tick_params(axis='both', which='major', labelsize=plot_dict["fontsize"]-3)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.legend(frameon=False, fontsize=plot_dict["fontsize"]-4)
	plt.tight_layout()

	if (plot_dict["shouldSave"]):
		plt.savefig(saveFile+".pdf")
		plt.savefig(saveFile+".jpg")


def main(args):

	results_path = os.path.join("pristine_model_ensemble_analysis_%s_branches_%s_%s.csv"%(args.n_branches, args.model_name, 
		args.dataset_name))

	savePath = os.path.join("plots", "accuracy", "%s_branches"%(args.n_branches))

	if(not os.path.exists(savePath)):
		os.makedirs(savePath)


	df = pd.read_csv(results_path)
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

	blur_levels = [0] + config.distortion_level_dict["gaussian_blur"]
	noise_levels = [0] + config.distortion_level_dict["gaussian_noise"]

	for threshold in config.threshold_list:
		for n_branch_edge in range(1, args.n_branches+1):

			plotAccuracyDistortionLevel(df, threshold, n_branch_edge, blur_levels, config.plot_dict, savePath, distortion_type="gaussian_blur")

			plotAccuracyDistortionLevel(df, threshold, n_branch_edge, noise_levels, config.plot_dict, savePath, distortion_type="gaussian_noise")

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
