import numpy as np
import pandas as pd
import argparse, config, os, sys, torch


def extractData(df, distortion_lvl, distortion_type_data):

	df = df[df.distortion_type_data=="pristine"] if (distortion_lvl==0) else df[df.distortion_type_data==distortion_type_data]

	df = df[df.distortion_lvl == distortion_lvl]

	return df

def compute_acc_early_exit(df, distortion_lvl, n_branches_edge, n_branches_total, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data) 

	numexits, correct_list = np.zeros(n_branches_edge), np.zeros(n_branches_edge)
	n_samples = len(df)

	remaining_data = df

	for i in range(1, n_branches_edge+1):	 

		if (i == n_branches_total):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			early_exit_samples = remaining_data["conf_branch_%s"%(i)] >= threshold

		numexits[i-1] = remaining_data[early_exit_samples]["conf_branch_%s"%(i)].count()
		correct_list[i-1] = remaining_data[early_exit_samples]["correct_branch_%s"%(i)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	acc_device = sum(correct_list)/sum(numexits)

	return acc_device


def compute_acc_backbone(df, distortion_lvl, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)
	
	acc = sum(df.correct.values)/len(df.correct.values)

	return acc


def compute_acc_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_branches_total, threshold, distortion_type_data):


	df_edge = df_ee[df_ee["ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold]

	nr_correct = sum(df_edge["ensemble_conf_branch_%s"%(n_branches_edge)].values)

	nr_samples = len(df_edge["ensemble_conf_branch_%s"%(n_branches_edge)].values)

	ensemble_edge_acc = nr_correct/nr_samples

	return ensemble_edge_acc

def extract_accuracy_edge(df_backbone, df_ee, n_branches_edge, n_exits, threshold, distortion_levels, distortion_type_data):

	acc_ee_list, acc_backbone_list, acc_ensemble_edge_list = [], [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branches_edge, distortion_lvl))

		acc_backbone = compute_acc_backbone(df_backbone, distortion_lvl, distortion_type_data)
		acc_ee = compute_acc_early_exit(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)
		acc_ensemble_edge = compute_acc_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)
		acc_ee_list.append(acc_ee), acc_backbone_list.append(acc_backbone), acc_ensemble_edge_list.append(acc_ensemble_edge)


def exp_ensemble_analysis(args, df_backbone, df_ee, distortion_type):

	distortion_levels = [0] + config.distortion_level_dict[distortion_type]
	n_exits = args.n_branches + 1
	
	for threshold in config.threshold_list:

		for n_branch in range(1, n_exits+1):

			#edge_prob_dict = extract_early_classification(df_ee, n_branch, args.n_branches, threshold, distortion_levels)
			acc_edge_dict = extract_accuracy_edge(df_backbone, df_ee, n_branch, n_exits, threshold, distortion_levels, 
				distortion_type)

			#plotDistortedEarlyClassification(edge_prob_dict, n_branch, args.distortion_type_data)

			#plotDistortedEdgeAccuracy(edge_prob_dict, n_branch, args.distortion_type_data)





def main(args):

	ee_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_%s_branches_id_%s_final.csv"%(args.n_branches, args.model_id))


	backbone_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_backbone_id_%s.csv"%(args.model_id))

	df_ee = pd.read_csv(ee_data_path)
	df_ee = df_ee.loc[:, ~df_ee.columns.str.contains('^Unnamed')] 

	df_backbone = pd.read_csv(backbone_data_path)
	df_backbone = df_backbone.loc[:, ~df_backbone.columns.str.contains('^Unnamed')]

	df_ee = df_ee[df_ee.distortion_type_model == args.distortion_type_model]
	df_backbone = df_backbone[df_backbone.distortion_type_model == args.distortion_type_model]


	exp_ensemble_analysis(args, df_backbone, df_ee, distortion_type="gaussian_blur")
	exp_ensemble_analysis(args, df_backbone, df_ee, distortion_type="gaussian_noise")





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
