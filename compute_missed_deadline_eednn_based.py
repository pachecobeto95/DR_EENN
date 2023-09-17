import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, argparse, os, sys, random, logging, config, math
#import torch, torchvision, utils
import scipy.stats as st


def chunker(df, batch_size):
	return [df[pos:pos + batch_size] for pos in range(0, len(df), batch_size)]

def compute_confidence_interval(outage_rounds, confidence=0.95):
	#return st.norm.interval(alpha=0.95, loc=np.mean(data), scale=st.sem(data))
	conf_interval = list(st.norm.interval(confidence, loc=np.mean(outage_rounds), scale=st.sem(outage_rounds)))
	
	if(math.isnan(conf_interval[0])):
		conf_interval[0] = 0

	if(math.isnan(conf_interval[1])):
		conf_interval[1] = 0

	return conf_interval


def computeInferenceTime(df_inf_time, threshold, n_branches, inf_mode):

	df_inf_time = df_inf_time[(df_inf_time.p_tar == threshold) & (df_inf_time["mode"] == inf_mode)]

	return df_inf_time.inference_time.mean()


def computeOverallAccuracy(df_batch, threshold, n_branches, inf_mode):

	n_samples = len(df_batch)
	mode = "" if(inf_mode == "eednn") else "ensemble_"
	n_exits = n_branches + 1
	numexits = np.zeros(n_exits)
	correct_list = np.zeros(n_exits)

	remaining_data = df_batch

	for i in range(1, n_exits+1):
		current_n_samples = len(remaining_data)

		if (i == n_exits):
			early_exit_samples = np.ones(current_n_samples, dtype=bool)
		else:
			early_exit_samples = remaining_data["%sconf_branch_%s"%(mode, i)] >= threshold

		numexits[i-1] = remaining_data[early_exit_samples]["%sconf_branch_%s"%(mode, i)].count()
		correct_list[i-1] = remaining_data[early_exit_samples]["%scorrect_branch_%s"%(mode, i)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	overall_acc = sum(correct_list)/n_samples
	#print(overall_acc)
	#sys.exit()

	return overall_acc


def computeMissedDeadlineProb(df_batches, df_inf_time, threshold, t_tar, n_branches, inf_mode):
	
	missed_deadline = 0
	
	for df_batch in df_batches:
		overall_acc = computeOverallAccuracy(df_batch, threshold, n_branches, inf_mode)
		inference_time = computeInferenceTime(df_inf_time, threshold, n_branches, inf_mode)

		missed_deadline += 1 if((overall_acc < threshold) or (inference_time > t_tar)) else 0

	missed_deadline_prob = float(missed_deadline)/len(df_batches)
	return missed_deadline_prob

def computeAvgMissedDeadlineProb(df, df_inf_time, threshold, t_tar, n_branches, n_rounds, n_batches, inf_mode):

	missed_deadline_rounds = []

	for n_round in range(n_rounds):
		#print("Number of Rounds: %s"%(n_round))

		df = df.sample(frac=1)
		df_batches = chunker(df, batch_size=n_batches)

		df_inf_time = df_inf_time.sample(frac=1)
		df_batches_inf_time = chunker(df_inf_time, batch_size=n_batches)

		print(type(df_batches_inf_time))


		missed_deadline_prob = computeMissedDeadlineProb(df_batches, df_batches_inf_time, threshold, t_tar, n_branches, inf_mode)

		missed_deadline_rounds.append(missed_deadline_prob)

	missed_deadline_rounds = np.array(missed_deadline_rounds)

	avg_missed_deadline = np.mean(missed_deadline_rounds)

	ic_missed_deadline = compute_confidence_interval(missed_deadline_rounds)

	return avg_missed_deadline, ic_missed_deadline[0], ic_missed_deadline[1] 


def getMissedDeadlineProbThreshold(df, df_inf_time, threshold, t_tar, distortion_lvl_list, n_branches, n_rounds, n_batches, inf_mode, dist_type_data):
	
	avg_missed_deadline_list, bottom_ic_md_list, upper_ic_md_list = [], [], []

	for distortion_lvl in distortion_lvl_list:
		df_dist_lvl = df[df.distortion_lvl == distortion_lvl]
		df_inf_time_dist_lvl = df_inf_time[df_inf_time.distortion_lvl == distortion_lvl]

		avg_missed_deadline, bottom_ic_md, upper_ic_md = computeAvgMissedDeadlineProb(df_dist_lvl, df_inf_time, threshold, 
			t_tar, n_branches, n_rounds, n_batches, inf_mode)

		avg_missed_deadline_list.append(avg_missed_deadline), bottom_ic_md_list.append(bottom_ic_md)
		upper_ic_md_list.append(upper_ic_md)

	result_dict = {"avg_missed_deadline": avg_missed_deadline_list, 
	"bottom_ic_missed_deadline": bottom_ic_md_list, 
	"upper_ic_missed_deadline": upper_ic_md_list, "threshold": len(avg_missed_deadline_list)*[threshold], 
	"t_tar": len(avg_missed_deadline_list)*[t_tar], "distortion_lvl": distortion_lvl_list, 
	"distortion_type_data": len(avg_missed_deadline_list)*[dist_type_data], 
	"n_rounds": len(avg_missed_deadline_list)*[n_rounds], "n_batches": len(avg_missed_deadline_list)*[n_batches],
	"inf_mode": len(avg_missed_deadline_list)*[inf_mode]}

	return result_dict



def save_missed_deadline_results(results, savePath):
	df = pd.DataFrame(results)
	df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))


def main(args):

	#missed_deadline_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023", 
	#	"missed_deadline_prob_%s_branches_id_%s.csv"%(args.n_branches, args.model_id))
	
	#inference_data_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023",  
	#	"inference_data_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	#inference_time_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023",  
	#	"inference_time_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	missed_deadline_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet", 
		"missed_deadline_prob_%s_branches_id_%s_final.csv"%(args.n_branches, args.model_id))
	
	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet",  
		"inference_data_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	inference_time_path = os.path.join(config.DIR_NAME, "inference_time.csv")

	df_inf_data = pd.read_csv(inference_data_path)
	df_inf_time = pd.read_csv(inference_time_path)

	threshold_list = [0.7, 0.8, 0.9]
	#threshold_list = [0.83, 0.85]
	t_tar_list = np.arange(config.t_tar_start, config.t_tar_end, config.t_tar_step)

	df_blur = df_inf_data[df_inf_data.distortion_type_data == "gaussian_blur"]

	distortion_lvl_list = df_inf_time.distortion_lvl.unique()


	for threshold in threshold_list:
		for t_tar in t_tar_list:
			print("Ttar: %s, Threshold: %s"%(t_tar, threshold))
			
			blur_missed_deadline = getMissedDeadlineProbThreshold(df_blur, df_inf_time, threshold, t_tar, 
				distortion_lvl_list, args.n_branches, args.n_rounds, args.n_batches, args.inf_mode, dist_type_data="gaussian_blur")
			
			#save_missed_deadline_results(pristine_missed_deadline, missed_deadline_path)
			save_missed_deadline_results(blur_missed_deadline, missed_deadline_path)


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
	parser.add_argument('--inf_mode', type=str, choices=["eednn", "ensemble"])


	args = parser.parse_args()
	main(args)
