import numpy as np
import pandas as pd
import argparse, config, os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker


def plotInfTimeDistortionLevel(df, threshold, n_branches_edge, distortion_levels, plot_dict, savePath, distortion_type):

	n_samples = 10000

	fig, ax = plt.subplots()

	df = df[(df.n_branches_edge == n_branches_edge) & (df.threshold == threshold) & (df.distortion_type_data==distortion_type)]

	saveFile = os.path.join(savePath, "%s_inference_time_%s_branches_threshold_%s"%(distortion_type, n_branches_edge, threshold))

	inf_time_backbone, std_inf_time_backbone = df.inf_time_backbone.values[:-1], df.std_inf_time_backbone.values[:-1]
	inf_time_ee, std_inf_time_ee = df.inf_time_ee.values[:-1], df.std_inf_time_ee.values[:-1]
	inf_time_ensemble, std_inf_time_ensemble = df.inf_time_ensemble.values[:-1], df.std_inf_time_ensemble.values[:-1]

	#distortion_levels = np.delete(distortion_levels[:-1], [1])
	distortion_levels = distortion_levels[:-1]
	#inf_time_backbone, std_inf_time_backbone = np.delete(inf_time_backbone, [1]), np.delete(std_inf_time_backbone, [1])
	#inf_time_ee, std_inf_time_ee = np.delete(inf_time_ee, [1]), np.delete(std_inf_time_ee, [1])
	
	#print(inf_time_ensemble, std_inf_time_ensemble)
	#inf_time_ensemble, std_inf_time_ensemble = np.delete(inf_time_ensemble, [1]), np.delete(std_inf_time_ensemble, [1])


	ci_backbone = 1.96 * std_inf_time_backbone/np.sqrt(n_samples)
	ci_ee = 1.96 * std_inf_time_ee/np.sqrt(n_samples)
	ci_ensemble = 1.96 * std_inf_time_ensemble/np.sqrt(n_samples)

	plt.plot(distortion_levels, inf_time_backbone, label="DNN Convencional", 
		marker=plot_dict["marker"][0], linestyle=plot_dict["line_style"][0], color=plot_dict["color"][0])

	ax.fill_between(distortion_levels, (inf_time_backbone-ci_backbone), (inf_time_backbone+ci_backbone), color=plot_dict["color"][0], alpha=.1)


	plt.plot(distortion_levels, inf_time_ee, label="EE-DNN", 
		marker=plot_dict["marker"][1], linestyle=plot_dict["line_style"][1], color=plot_dict["color"][1])

	ax.fill_between(distortion_levels, (inf_time_ee-ci_ee), (inf_time_ee+ci_ee), color=plot_dict["color"][1], alpha=.1)


	plt.plot(distortion_levels, inf_time_ensemble, label="Comitê EE-DNN", 
		marker=plot_dict["marker"][2], linestyle=plot_dict["line_style"][2], color=plot_dict["color"][2])

	ax.fill_between(distortion_levels, (inf_time_ensemble-ci_ensemble), (inf_time_ensemble+ci_ensemble), color=plot_dict["color"][2], alpha=.1)


	#plt.plot(distortion_levels, df.overall_acc_naive_ensemble.values, label="Comitê EE-DNN 2", 
	#	marker=plot_dict["marker"][3], linestyle=plot_dict["line_style"][3], color=plot_dict["color"][3])


	plt.xlabel(plot_dict["x_axis"][distortion_type], fontsize=plot_dict["fontsize"])
	#plt.ylabel("Overall Accuracy", fontsize=plot_dict["fontsize"])
	plt.ylabel("Tempo de Inferência (ms)", fontsize=plot_dict["fontsize"])
	ax.tick_params(axis='both', which='major', labelsize=plot_dict["fontsize"]-3)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.legend(frameon=False, fontsize=plot_dict["fontsize"]-4)
	plt.tight_layout()

	if (plot_dict["shouldSave"]):
		plt.savefig(saveFile+".pdf")
		plt.savefig(saveFile+".jpg")


def plotFlopDistortionLevel(df, threshold, n_branches_edge, distortion_levels, plot_dict, savePath, distortion_type):

	fig, ax = plt.subplots()

	formatter = ticker.ScalarFormatter(useMathText=True)

	df = df[(df.n_branches_edge == n_branches_edge) & (df.threshold == threshold) & (df.distortion_type_data==distortion_type)]

	saveFile = os.path.join(savePath, "%s_flop_%s_branches_threshold_%s"%(distortion_type, n_branches_edge, threshold))


	plt.plot(distortion_levels, df.flops_backbone.values, label="DNN Convencional", 
		marker=plot_dict["marker"][0], linestyle=plot_dict["line_style"][0], color=plot_dict["color"][0])


	plt.plot(distortion_levels, df.flops_ee.values, label="EE-DNN", 
		marker=plot_dict["marker"][1], linestyle=plot_dict["line_style"][1], color=plot_dict["color"][1])


	plt.plot(distortion_levels, df.flops_ensemble.values, label="Comitê EE-DNN", 
		marker=plot_dict["marker"][2], linestyle=plot_dict["line_style"][2], color=plot_dict["color"][2])

	#plt.plot(distortion_levels, df.overall_acc_naive_ensemble.values, label="Comitê EE-DNN 2", 
	#	marker=plot_dict["marker"][3], linestyle=plot_dict["line_style"][3], color=plot_dict["color"][3])


	plt.xlabel(plot_dict["x_axis"][distortion_type], fontsize=plot_dict["fontsize"])
	#plt.ylabel("Overall Accuracy", fontsize=plot_dict["fontsize"])
	plt.ylabel("Flops", fontsize=plot_dict["fontsize"])
	ax.tick_params(axis='both', which='major', labelsize=plot_dict["fontsize"]-3)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	plt.legend(frameon=False, fontsize=plot_dict["fontsize"]-4)
	formatter.set_scientific(True) 
	formatter.set_powerlimits((-1,1)) 
	ax.yaxis.set_major_formatter(formatter) 
	plt.tight_layout()

	if (plot_dict["shouldSave"]):
		plt.savefig(saveFile+".pdf")
		plt.savefig(saveFile+".jpg")


def main(args):

	results_path = os.path.join("pristine_model_ensemble_analysis_%s_branches_%s_%s_final_final_final.csv"%(args.n_branches, args.model_name, 
		args.dataset_name))

	inf_time_savePath = os.path.join("plots", "inference_time", "%s_branches_final_final"%(args.n_branches))

	flops_savePath = os.path.join("plots", "flops", "%s_branches_final_final"%(args.n_branches))

	if(not os.path.exists(inf_time_savePath)):
		os.makedirs(inf_time_savePath)


	if(not os.path.exists(flops_savePath)):
		os.makedirs(flops_savePath)


	df = pd.read_csv(results_path)
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

	blur_levels = [0] + config.distortion_level_dict["gaussian_blur"]
	noise_levels = [0] + config.distortion_level_dict["gaussian_noise"]

	for threshold in config.threshold_list:
		for n_branch_edge in range(1, args.n_branches+1):

			plotInfTimeDistortionLevel(df, threshold, n_branch_edge, blur_levels, config.plot_dict, inf_time_savePath, distortion_type="gaussian_blur")

			#plotFlopDistortionLevel(df, threshold, n_branch_edge, blur_levels, config.plot_dict, flops_savePath, distortion_type="gaussian_blur")

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
