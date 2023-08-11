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

def computeOnDeviceAccuracy(df_batch, threshold, n_branches, inf_mode):

	numexits = np.zeros(n_branches)
	correct_list = np.zeros(n_branches)
	n_samples = len(df_batch)
	mode = "" if(inf_mode == "eednn") else "ensemble_"


	remaining_data = df_batch
	#sys.exit()

	for i in range(1, n_branches+1):
		current_n_samples = len(remaining_data)

		#if (i == n_exits):
		#	early_exit_samples = np.ones(current_n_samples, dtype=bool)
		#else:
		early_exit_samples = remaining_data["%sconf_branch_%s"%(mode, i)] >= threshold

		numexits[i-1] = remaining_data[early_exit_samples]["%sconf_branch_%s"%(mode, i)].count()
		correct_list[i-1] = remaining_data[early_exit_samples]["%scorrect_branch_%s"%(mode, i)].sum()

		remaining_data = remaining_data[~early_exit_samples]

	acc_device = sum(correct_list)/sum(numexits)

	return acc_device


def computeInferenceOutageProb(df_batches, threshold, n_branches, inf_mode):
	outage = 0
	for df_batch in df_batches:
		on_device_acc = computeOnDeviceAccuracy(df_batch, threshold, n_branches, inf_mode)
		outage += 1 if(on_device_acc < threshold) else 0

	outage_prob = float(outage)/len(df_batches)
	return outage_prob


def computeAvgInferenceOutageProb(df, threshold, n_branches, n_rounds, n_batches, inf_mode):

	outage_rounds = []

	for n_round in range(n_rounds):
		#print("Number of Rounds: %s"%(n_round))

		df = df.sample(frac=1)
		df_batches = chunker(df, batch_size=n_batches)

		outage_prob = computeInferenceOutageProb(df_batches, threshold, n_branches, inf_mode)

		outage_rounds.append(outage_prob)

	outage_rounds = np.array(outage_rounds)

	avg_outage = np.mean(outage_rounds)

	ic_outage = compute_confidence_interval(outage_rounds)

	return avg_outage, ic_outage[0], ic_outage[1] 


def getInfOutageProbThreshold(df, threshold, n_branches, n_rounds, n_batches, inf_mode, dist_type_data):
	
	avg_outage_list, bottom_ic_outage_list, upper_ic_outage_list = [], [], []

	for distortion_lvl in df.distortion_lvl.unique():
		df_dist_lvl = df[df.distortion_lvl == distortion_lvl]

		avg_outage, bottom_ic_outage, upper_ic_outage = computeAvgInferenceOutageProb(df, threshold, n_branches, n_rounds, 
			n_batches, inf_mode)

		avg_outage_list.append(avg_outage), bottom_ic_outage_list.append(bottom_ic_outage)
		upper_ic_outage_list.append(upper_ic_outage)

	result_dict = {"avg_outage": avg_outage_list, "bottom_ic_outage": bottom_ic_outage_list, 
	"upper_ic_outage": upper_ic_outage_list, "threshold": len(avg_outage_list)*[threshold], 
	"distortion_lvl": df.distortion_lvl.unique(), 
	"distortion_type_data": len(avg_outage_list)*[dist_type_data], 
	"n_rounds": len(avg_outage_list)*[n_rounds], "n_batches": len(avg_outage_list)*[n_batches]}

	return result_dict

def save_outage_results(outage, savePath):
	df = pd.DataFrame(outage)
	df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))


def main(args):

	#edge_inf_outage_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023", 
	#	"edge_inf_outage_prob_%s_branches_id_%s.csv"%(args.n_branches, args.model_id))
	
	#inference_data_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023",  
	#	"inference_data_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))

	edge_inf_outage_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet", 
		"edge_inf_outage_prob_%s_branches_id_%s.csv"%(args.n_branches, args.model_id))
	
	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet",  
		"inference_data_%s_branches_id_%s_final_final.csv"%(args.n_branches, args.model_id))


	threshold_list = np.arange(config.threshold_start, config.threshold_end, config.threshold_step)

	df_inf_data = pd.read_csv(inference_data_path)

	df_pristine = df_inf_data[df_inf_data.distortion_type_data == "pristine"]
	df_blur = df_inf_data[df_inf_data.distortion_type_data == "gaussian_blur"]
	df_noise = df_inf_data[df_inf_data.distortion_type_data == "gaussian_noise"]

	for threshold in threshold_list:
		#print("Threshold: %s"%(threshold))

		pristine_outage = getInfOutageProbThreshold(df_pristine, threshold, args.n_branches, args.n_rounds, 
			args.n_batches, args.inf_mode, dist_type_data="pristine")
		blur_outage = getInfOutageProbThreshold(df_blur, threshold, args.n_branches, args.n_rounds, 
			args.n_batches, args.inf_mode, dist_type_data="gaussian_blur")
		
		save_outage_results(pristine_outage, edge_inf_outage_path)
		save_outage_results(blur_outage, edge_inf_outage_path)


		#noise_outage = getInfOutageProbThreshold(df_pristine, threshold, args.n_branches, args.n_rounds, args.n_batches)



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
