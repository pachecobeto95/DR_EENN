import numpy as np
import pandas as pd
import argparse, config, os, sys, torch, math


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

		remaining_data = remaining_data[~early_exit_samples]

	if (n_branches_edge != n_branches_total):
		correct += remaining_data["correct_branch_%s"%(n_branches_total)].sum()

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
	correct += df_cloud["correct_branch_%s"%(n_exits)].sum()


	ensemble_overall_acc = float(correct)/n_samples
	return ensemble_overall_acc



def compute_overall_acc_naive_ensemble(df, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)

	n_samples = len(df)
	correct = 0.0
	early_exit_samples = df["naive_ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold

	df_edge = df[early_exit_samples]
	df_cloud = df[~early_exit_samples]

	correct += df_edge["naive_ensemble_correct_branch_%s"%(n_branches_edge)].sum()
	correct += df_cloud["correct_branch_%s"%(n_exits)].sum()

	naive_ensemble_edge_acc = float(correct)/n_samples
	return naive_ensemble_edge_acc


def extract_accuracy_edge(df_backbone, df_ee, n_branches_edge, n_exits, threshold, distortion_levels, distortion_type_data):

	acc_ee_list, acc_backbone_list, acc_ensemble_edge_list, acc_naive_ensemble_edge_list = [], [], [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branches_edge, distortion_lvl))

		acc_backbone = compute_acc_backbone(df_backbone, distortion_lvl, distortion_type_data)
		acc_ee = compute_acc_early_exit(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)
		acc_ensemble_edge = compute_acc_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)
		acc_naive_ensemble_edge = compute_acc_naive_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data)

		acc_ee_list.append(acc_ee), acc_backbone_list.append(acc_backbone), acc_ensemble_edge_list.append(acc_ensemble_edge)
		acc_naive_ensemble_edge_list.append(acc_naive_ensemble_edge)


	return {"acc_ee": acc_ee_list, "acc_backbone": acc_backbone_list, "acc_ensemble": acc_ensemble_edge_list, "acc_naive_ensemble": acc_naive_ensemble_edge_list}


def extract_overall_accuracy(df_backbone, df_ee, n_branches_edge, n_exits, threshold, distortion_levels, distortion_type):
	acc_ee_list, acc_backbone_list, acc_ensemble_edge_list, overall_acc_naive_ensemble_edge_list = [], [], [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branches_edge, distortion_lvl))

		acc_backbone = compute_acc_backbone(df_backbone, distortion_lvl, distortion_type)

		overall_acc_ee = compute_overall_acc_early_exit(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type)
		overall_acc_ensemble_edge = compute_overall_acc_ensemble_ee_edge(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type)

		overall_acc_naive_ensemble = compute_overall_acc_naive_ensemble(df_ee, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type)


		acc_ee_list.append(overall_acc_ee), acc_backbone_list.append(acc_backbone), acc_ensemble_edge_list.append(overall_acc_ensemble_edge)
		overall_acc_naive_ensemble_edge_list.append(overall_acc_naive_ensemble)

	return {"overall_acc_ee": acc_ee_list, "acc_backbone": acc_backbone_list, "overall_acc_ensemble": acc_ensemble_edge_list,
	"overall_acc_naive_ensemble": overall_acc_naive_ensemble_edge_list}



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

	n_samples = len(df)

	early_exit_samples = df["ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold

	df_edge = df[early_exit_samples]

	n_early_exit = len(df_edge)

	return n_early_exit/n_samples


def compute_naive_ensemble_early_prob(df, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type_data):

	df = extractData(df, distortion_lvl, distortion_type_data)

	n_samples = len(df)

	early_exit_samples = df["naive_ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold

	df_edge = df[early_exit_samples]

	n_early_exit = len(df_edge)

	return n_early_exit/n_samples


def extract_early_classification(df_ee, n_branch_edge, n_exits, threshold, distortion_levels, distortion_type_data):

	early_prob_ee_list, early_prob_ensemble_edge_list, early_prob_naive_ensemble_edge_list = [], [], []

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branch_edge, distortion_lvl))

		ee_early_prob = compute_early_prob(df_ee, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type_data)

		ensemble_early_prob = compute_ensemble_early_prob(df_ee, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type_data)

		naive_ensemble_early_prob = compute_naive_ensemble_early_prob(df_ee, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type_data)

		early_prob_ee_list.append(ee_early_prob), early_prob_ensemble_edge_list.append(ensemble_early_prob)
		early_prob_naive_ensemble_edge_list.append(naive_ensemble_early_prob)

	return {"ee_early_prob": early_prob_ee_list, "ensemble_early_prob": early_prob_ensemble_edge_list, "naive_ensemble_early_prob": early_prob_naive_ensemble_edge_list}

def extract_inf_time(df_backbone, df_ee, df_ee_inf_time, df_backbone_inf_time, n_branch_edge, n_exits, threshold, distortion_levels, distortion_type):
	inf_time_backbone_list, inf_time_ee_list, inf_time_ensemble_list, inf_time_naive_ensemble_list  = [], [], [], []

	std_inf_time_backbone_list, std_inf_time_ee_list, std_inf_time_ensemble_list, std_inf_time_naive_ensemble_list  = [], [], [], []

	flops_backbone_list, flops_ee_list, flops_ensemble_list, flops_naive_ensemble_list = [], [], [], []	

	for distortion_lvl in distortion_levels:
		print("Threshold: %s, Nr of branches at the Edge: %s, Distortion Lvl: %s"%(threshold, n_branch_edge, distortion_lvl))

		inf_time_backbone, std_inf_time_backbone, flops_backbone = compute_inf_time_backbone(df_backbone, df_backbone_inf_time, distortion_lvl, distortion_type)

		inf_time_ee, std_inf_time_ee, flops_ee = compute_inf_time_ee(df_ee, df_ee_inf_time, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type)
		inf_time_ensemble, std_inf_time_ensemble, flops_ensemble = compute_inf_time_ensemble(df_ee, df_ee_inf_time, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type)
		inf_time_naive_ensemble, std_inf_time_naive_ensemble, flops_naive_ensemble = compute_inf_time_naive_ensemble(df_ee, df_ee_inf_time, distortion_lvl, n_branch_edge, n_exits, threshold, distortion_type)

		inf_time_ee_list.append(inf_time_ee), inf_time_backbone_list.append(inf_time_backbone)
		inf_time_ensemble_list.append(inf_time_ensemble), inf_time_naive_ensemble_list.append(inf_time_naive_ensemble)

		flops_backbone_list.append(flops_backbone), flops_ee_list.append(flops_ee), flops_ensemble_list.append(flops_ensemble)
		flops_naive_ensemble_list.append(flops_naive_ensemble)

		std_inf_time_backbone_list.append(std_inf_time_backbone), std_inf_time_ee_list.append(std_inf_time_ee), std_inf_time_ensemble_list.append(std_inf_time_ensemble)
		std_inf_time_naive_ensemble_list.append(std_inf_time_naive_ensemble)

	return {"inf_time_ee": inf_time_ee_list, "inf_time_backbone": inf_time_backbone_list, "inf_time_ensemble": inf_time_ensemble_list, 
	"inf_time_naive_ensemble": inf_time_naive_ensemble_list, 
	"flops_backbone": flops_backbone_list, "flops_ee": flops_ee_list, "flops_ensemble": flops_ensemble_list, 
	"flops_naive_ensemble": flops_naive_ensemble_list,
	"std_inf_time_backbone": std_inf_time_backbone_list,
	"std_inf_time_ee": std_inf_time_ee_list,
	"std_inf_time_ensemble": std_inf_time_ensemble_list,
	"std_inf_time_naive_ensemble": std_inf_time_naive_ensemble_list}

def compute_inf_time_backbone(df, df_backbone_inf_time, distortion_lvl, distortion_type):
	df = extractData(df, distortion_lvl, distortion_type)

	inf_time_backbone = df_backbone_inf_time.inference_time.mean()

	#print(df_backbone_inf_time.inference_time.values)

	#print("Backbone: %s"%(inf_time_backbone))

	flops_backbone = df.flops.mean()

	inf_time_backbone_std = df_backbone_inf_time.inference_time.std()

	return inf_time_backbone, inf_time_backbone_std, flops_backbone

def compute_standard_dev(std_inf_time_list):
	cumulative_std_inf_time = 0

	for std_inf_time in std_inf_time_list:
		cumulative_std_inf_time += 	std_inf_time**2
	
	return math.sqrt(cumulative_std_inf_time)	

def compute_inf_time_ee(df, df_ee_inf_time, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type):

	df = extractData(df, distortion_lvl, distortion_type)

	numexits = np.zeros(n_branches_edge)
	n_samples = len(df)
	remaining_data = df
	inference_time, flops = 0.0, 0.0
	std_inf_time_list = []

	for i in range(1, n_branches_edge+1):
		current_n_samples = len(remaining_data)
		early_exit_samples = remaining_data["conf_branch_%s"%(i)] >= threshold
		numexits[i-1] = remaining_data[early_exit_samples]["conf_branch_%s"%(i)].count()
		remaining_data = remaining_data[~early_exit_samples]

		#print("EE: %s"%(df_ee_inf_time["inference_time_branches_%s"%(i)].mean()))
		inference_time += numexits[i-1]*df_ee_inf_time["inference_time_branches_%s"%(i)].mean()
		flops += numexits[i-1]*df["flops_branches_%s"%(i)].mean()

		std_inf_time_list.append(df_ee_inf_time["inference_time_branches_%s"%(i)].std())

	#print("EE: %s"%(df_ee_inf_time["inference_time_branches_%s"%(n_exits)].mean()))
	n_samples_cloud = remaining_data["conf_branch_%s"%(n_exits)].count()
	inference_time += n_samples_cloud*df_ee_inf_time["inference_time_branches_%s"%(n_exits)].mean()
	flops += n_samples_cloud*df["flops_branches_%s"%(n_exits)].mean()

	std_inf_time_list.append(df_ee_inf_time["inference_time_branches_%s"%(n_exits)].std())

	avg_inf_time = inference_time/n_samples
	avg_flops = flops/n_samples

	std_inf_time = compute_standard_dev(std_inf_time_list)

	return avg_inf_time, std_inf_time, avg_flops

def compute_inf_time_ensemble(df, df_ee_inf_time, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type):

	std_inf_time_list = []

	df = extractData(df, distortion_lvl, distortion_type)

	n_samples = len(df)

	early_exit_samples = df["ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold


	df_edge = df[early_exit_samples]

	n_early_exit = len(df_edge)

	#print(n_branches_edge, n_early_exit)

	inference_time = n_early_exit*df_ee_inf_time["inference_time_branches_%s"%(n_branches_edge)].mean()
	flops = n_early_exit*df["flops_branches_%s"%(n_branches_edge)].mean()
	std_inf_time_list.append(df_ee_inf_time["inference_time_branches_%s"%(n_branches_edge)].std())

	inference_time += (n_samples-n_early_exit)*df_ee_inf_time["inference_time_branches_%s"%(n_exits)].mean()
	flops += (n_samples-n_early_exit)*df["flops_branches_%s"%(n_exits)].mean()
	std_inf_time_list.append(df_ee_inf_time["inference_time_branches_%s"%(n_exits)].std())

	avg_inf_time = inference_time/n_samples
	avg_flops = flops/n_samples

	std_inf_time = compute_standard_dev(std_inf_time_list)

	return avg_inf_time, std_inf_time, avg_flops

def compute_inf_time_naive_ensemble(df, df_ee_inf_time, distortion_lvl, n_branches_edge, n_exits, threshold, distortion_type):

	std_inf_time_list = []

	df = extractData(df, distortion_lvl, distortion_type)

	n_samples = len(df)

	early_exit_samples = df["naive_ensemble_conf_branch_%s"%(n_branches_edge)] >= threshold

	df_edge = df[early_exit_samples]

	n_early_exit = len(df_edge)


	print(n_branches_edge, n_early_exit)


	inference_time = n_early_exit*df_ee_inf_time["inference_time_branches_%s"%(n_branches_edge)].mean()
	flops = n_early_exit*df["flops_branches_%s"%(n_branches_edge)].mean()
	std_inf_time_list.append(df_ee_inf_time["inference_time_branches_%s"%(n_branches_edge)].std())

	inference_time += (n_samples-n_early_exit)*df_ee_inf_time["inference_time_branches_%s"%(n_exits)].mean()
	flops += (n_samples-n_early_exit)*df["flops_branches_%s"%(n_exits)].mean()
	std_inf_time_list.append(df_ee_inf_time["inference_time_branches_%s"%(n_exits)].std())

	avg_inf_time = inference_time/n_samples
	avg_flops = flops/n_samples

	std_inf_time = compute_standard_dev(std_inf_time_list)

	return avg_inf_time, std_inf_time, avg_flops


def exp_ensemble_analysis(args, df_backbone, df_ee, df_ee_inf_time, df_backbone_inf_time, save_path, distortion_type):

	distortion_levels = [0] + config.distortion_level_dict[distortion_type]
	n_exits = args.n_branches + 1

	for threshold in config.threshold_list:

		for n_branch_edge in range(1, n_exits+1):

			edge_prob_dict = extract_early_classification(df_ee, n_branch_edge, n_exits, threshold, distortion_levels, distortion_type)
			#acc_edge_dict = extract_accuracy_edge(df_backbone, df_ee, n_branch_edge, n_exits, threshold, distortion_levels, 
			#	distortion_type)
			acc_overall_dict = extract_overall_accuracy(df_backbone, df_ee, n_branch_edge, n_exits, threshold, distortion_levels, 
				distortion_type)

			inference_time_dict = extract_inf_time(df_backbone, df_ee, df_ee_inf_time, df_backbone_inf_time, n_branch_edge, n_exits, threshold, distortion_levels, distortion_type)


			save_results(acc_overall_dict, edge_prob_dict, inference_time_dict, distortion_levels, n_branch_edge, threshold, distortion_type, save_path)

def save_results(acc_edge_dict, edge_prob_dict, inference_time_dict, distortion_levels, n_branch, threshold, distortion_type_data, save_path):
	results_dict = {}
	results_dict.update(acc_edge_dict), results_dict.update(edge_prob_dict), results_dict.update(inference_time_dict)
	#results_dict.update(flops_dict)

	results_dict.update({"distortion_lvl": distortion_levels, "distortion_type_data": distortion_type_data, 
		"n_branches_edge": [n_branch]*len(distortion_levels), "threshold": len(distortion_levels)*[threshold]})

	df = pd.DataFrame(results_dict)
	df.to_csv(save_path, mode='a', header=not os.path.exists(save_path) )


def main(args):

	ee_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	backbone_data_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_data_backbone_id_%s_final_final.csv"%(args.model_id))

	ee_inf_time_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_time_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	backbone_inf_time_path = os.path.join(config.DIR_NAME, "inference_data", args.dataset_name, args.model_name, 
		"inference_time_backbone_id_%s_final_final.csv"%(args.model_id))

	save_results_dir = os.path.join(config.DIR_NAME, "ensemble_analysis_results")

	if(not os.path.exists(save_results_dir)):
		os.makedirs(save_results_dir)

	df_ee = pd.read_csv(ee_data_path)
	df_ee = df_ee.loc[:, ~df_ee.columns.str.contains('^Unnamed')]

	df_backbone = pd.read_csv(backbone_data_path)
	df_backbone = df_backbone.loc[:, ~df_backbone.columns.str.contains('^Unnamed')]

	df_ee = df_ee[df_ee.distortion_type_model == args.distortion_type_model]
	df_backbone = df_backbone[df_backbone.distortion_type_model == args.distortion_type_model]


	df_ee_inf_time = pd.read_csv(ee_inf_time_path)
	df_ee_inf_time = df_ee_inf_time.loc[:, ~df_ee_inf_time.columns.str.contains('^Unnamed')]

	df_backbone_inf_time = pd.read_csv(backbone_inf_time_path)
	df_backbone_inf_time = df_backbone_inf_time.loc[:, ~df_backbone_inf_time.columns.str.contains('^Unnamed')]

	save_results_path = os.path.join(save_results_dir, 
		"%s_model_ensemble_analysis_%s_branches_%s_%s_final_final_final.csv"%(args.distortion_type_model, args.n_branches, args.model_name, args.dataset_name))


	exp_ensemble_analysis(args, df_backbone, df_ee, df_ee_inf_time, df_backbone_inf_time, save_results_path, distortion_type="gaussian_blur")
	#exp_ensemble_analysis(args, df_backbone, df_ee, save_results_path, distortion_type="gaussian_noise")


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
