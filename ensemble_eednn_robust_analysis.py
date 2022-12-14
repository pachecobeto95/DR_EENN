import numpy as np
import pandas as pd
import argparse, config, os, sys, torch


def extractData(df, distortion_lvl, distortion_type_data):

	df = df[df.distortion_type_data=="pristine"] if (distortion_lvl==0) else df[df.distortion_type_data==distortion_type_data]

	df = df[df.distortion_lvl == distortion_lvl]

	return df


def compute_overall_acc_early_exit(df, distortion_lvl, n_branches_edge, n_branches_total, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data) 

	correct = 0.0

	remaining_data = df
	n_samples = len(df)

	for i in range(1, n_branches_edge+1):	 
		current_n_samples = len(remaining_data)

		if (i == n_branches_total):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			early_exit_samples = remaining_data["conf_branch_%s"%(i)] >= threshold

		correct += remaining_data[early_exit_samples]["correct_branch_%s"%(i)].sum()
		print(remaining_data["conf_branch_%s"%(i)].mean(), len(early_exit_samples))

		remaining_data = remaining_data[~early_exit_samples]

	if (n_branches_edge != n_branches_total):
		correct += remaining_data[early_exit_samples]["correct_branch_%s"%(n_branches_total)].sum()

	overall_acc = float(correct)/n_samples
	return overall_acc

def compute_acc_early_exit(df, distortion_lvl, n_branches_edge, n_branches_total, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data) 

	numexits, correct_list = np.zeros(n_branches_edge), np.zeros(n_branches_edge)
	n_samples = len(df)

	remaining_data = df

	for i in range(1, n_branches_edge+1):	 
		current_n_samples = len(remaining_data)

		if (i == n_branches_total):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			early_exit_samples = remaining_data["conf_branch_%s"%(i)] >= threshold

		numexits[i-1] = remaining_data[early_exit_samples]["conf_branch_%s"%(i)].count()
		correct_list[i-1] = remaining_data[early_exit_samples]["correct_branch_%s"%(i)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	#print(numexits)
	acc_device = sum(correct_list)/sum(numexits) if(sum(numexits)>0) else 0

	return acc_device


def compute_acc_backbone(df, distortion_lvl, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)
	
	acc = sum(df.correct.values)/len(df.correct.values) if(len(df.correct.values)>0) else 0

	return acc

def compute_overall_acc_ensemble_ee_edge(df, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)
	n_samples = len(df)

	correct = 0

	early_exit_samples = df["ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold

	df_edge = df[early_exit_samples]
	df_cloud = df[~early_exit_samples]


	correct += df_edge["ensemble_correct_branch_%s"%(n_branches_edge)].sum()
	correct += df_cloud["ensemble_correct_branch_%s"%(n_exits)].sum()

	ensemble_overall_acc = float(correct)/n_samples
	return ensemble_overall_acc

def compute_acc_ensemble_ee_edge(df, distortion_lvl, n_branches_edge, n_branches_total, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)

	df_edge = df[df["ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold]

	nr_correct = sum(df_edge["ensemble_correct_branch_%s"%(n_branches_edge)].values)

	nr_samples = len(df_edge["ensemble_correct_branch_%s"%(n_branches_edge)].values)

	ensemble_edge_acc = nr_correct/nr_samples if(nr_samples>0) else 0

	return ensemble_edge_acc

def compute_acc_naive_ensemble_ee_edge(df, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)

	df_edge = df[df["naive_ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold]

	nr_correct = sum(df_edge["naive_ensemble_correct_branch_%s"%(n_branches_edge)].values)

	nr_samples = len(df_edge["naive_ensemble_correct_branch_%s"%(n_branches_edge)].values)

	naive_ensemble_edge_acc = nr_correct/float(nr_samples) if(nr_samples>0) else 0

	return naive_ensemble_edge_acc

def extract_accuracy_edge(df_backbone, df_ee, n_branches_edge, n_exits, threshold, distortion_levels, distortion_type_data):

	acc_ee_list, acc_backbone_list, acc_ensemble_edge_list, acc_naive_ensemble_edge_list = [], [], [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branches_edge, distortion_lvl))

		acc_backbone = compute_acc_backbone(df_backbone, distortion_lvl, distortion_type_data)
		acc_ee = compute_acc_early_exit(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)
		acc_ensemble_edge = compute_acc_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)
		acc_naive_ensemble_edge = compute_acc_naive_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)

		#save_results(acc_backbone, acc_ee, acc_ensemble_edge, distortion_lvl, n_branches_edge)
		acc_ee_list.append(acc_ee), acc_backbone_list.append(acc_backbone), acc_ensemble_edge_list.append(acc_ensemble_edge)
		acc_naive_ensemble_edge_list.append(acc_naive_ensemble_edge)


	return {"acc_ee": acc_ee_list, "acc_backbone": acc_backbone_list, "acc_ensemble": acc_ensemble_edge_list, "acc_naive_ensemble": acc_naive_ensemble_edge_list}


def extract_overall_accuracy(df_backbone, df_ee, n_branches_edge, n_exits, threshold, distortion_levels, distortion_type):
	acc_ee_list, acc_backbone_list, acc_ensemble_edge_list = [], [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branches_edge, distortion_lvl))

		acc_backbone = compute_acc_backbone(df_backbone, distortion_lvl, distortion_type)

		overall_acc_ee = compute_overall_acc_early_exit(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type)
		overall_acc_ensemble_edge = compute_overall_acc_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type)

		acc_ee_list.append(overall_acc_ee), acc_backbone_list.append(acc_backbone), acc_ensemble_edge_list.append(overall_acc_ensemble_edge)

	return {"overall_acc_ee": acc_ee_list, "acc_backbone": acc_backbone_list, "overall_acc_ensemble": acc_ensemble_edge_list}



def compute_early_prob(df, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)

	numexits = np.zeros(n_branches_edge)
	n_samples = len(df)
	remaining_data = df

	for i in range(1, n_branches_edge+1):
		current_n_samples = len(remaining_data)
		early_exit_samples = remaining_data["conf_branch_%s"%(i)] >= threshold
		numexits[i-1] = remaining_data[early_exit_samples]["conf_branch_%s"%(i)].count()
		remaining_data = remaining_data[~early_exit_samples]

	return sum(numexits)/n_samples


def compute_ensemble_early_prob(df, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)

	df_ensemble = df[df["ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold]

	n_early_exit = df_ensemble["ensemble_conf_branch_%s"%(n_branches_edge)].count()

	n_samples = len(df["ensemble_conf_branch_%s"%(n_branches_edge)].values)


	return n_early_exit/n_samples


def extract_early_classification(df_ee, n_branch_edge, n_exits, threshold, distortion_levels, distortion_type_data):

	early_prob_ee_list, early_prob_ensemble_edge_list = [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branch_edge, distortion_lvl))

		ee_early_prob = compute_early_prob(df_ee, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type_data)

		ensemble_early_prob = compute_ensemble_early_prob(df_ee, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type_data)

		early_prob_ee_list.append(ee_early_prob), early_prob_ensemble_edge_list.append(ensemble_early_prob)

	return {"ee_early_prob": early_prob_ee_list, "ensemble_early_prob": early_prob_ensemble_edge_list}

def exp_ensemble_analysis(args, df_backbone, df_ee, save_path, distortion_type):

	distortion_levels = [0] + config.distortion_level_dict[distortion_type]
	n_exits = args.n_branches + 1

	for threshold in config.threshold_list:

		for n_branch_edge in range(1, n_exits+1):

			edge_prob_dict = extract_early_classification(df_ee, n_branch_edge, n_exits, threshold, distortion_levels, distortion_type)
			#acc_edge_dict = extract_accuracy_edge(df_backbone, df_ee, n_branch_edge, n_exits, threshold, distortion_levels, 
			#	distortion_type)
			acc_overall_dict = extract_overall_accuracy(df_backbone, df_ee, n_branch_edge, n_exits, threshold, distortion_levels, 
				distortion_type)

			save_results(acc_overall_dict, edge_prob_dict, distortion_levels, n_branch_edge, threshold, distortion_type, save_path)


def save_results(acc_edge_dict, edge_prob_dict, distortion_levels, n_branch, threshold, distortion_type_data, save_path):
	results_dict = {}
	results_dict.update(acc_edge_dict), results_dict.update(edge_prob_dict) 
	results_dict.update({"distortion_lvl": distortion_levels, "distortion_type_data": distortion_type_data, 
		"n_branches_edge": [n_branch]*len(distortion_levels), "threshold": len(distortion_levels)*[threshold]})

	df = pd.DataFrame(results_dict)
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def main(args):

	ee_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_%s_branches_id_%s_final.csv"%(args.n_branches, args.model_id))


	backbone_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_backbone_id_%s.csv"%(args.model_id))

	save_results_dir = os.path.join(config.DIR_NAME, "ensemble_analysis_results")

	if(not os.path.exists(save_results_dir)):
		os.makedirs(save_results_dir)

	save_results_path = os.path.join(save_results_dir, 
		"%s_model_ensemble_analysis_%s_branches_%s_%s.csv"%(args.distortion_type_model, args.n_branches, args.model_name, args.dataset_name))

	df_ee = pd.read_csv(ee_data_path)
	df_ee = df_ee.loc[:, ~df_ee.columns.str.contains('^Unnamed')]

	df_backbone = pd.read_csv(backbone_data_path)
	df_backbone = df_backbone.loc[:, ~df_backbone.columns.str.contains('^Unnamed')]

	df_ee = df_ee[df_ee.distortion_type_model == args.distortion_type_model]
	df_backbone = df_backbone[df_backbone.distortion_type_model == args.distortion_type_model]

	exp_ensemble_analysis(args, df_backbone, df_ee, save_results_path, distortion_type="gaussian_blur")
	exp_ensemble_analysis(args, df_backbone, df_ee, save_results_path, distortion_type="gaussian_noise")


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
