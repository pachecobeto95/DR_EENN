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


def computeInferenceTime(df_batch, df_inf_time, threshold):

	#n_samples = len(df_batch)

	#avg_inference_time = df_inf_time.inference_time.mean()

	return np.mean(df_inf_time)

def computeOverallAccuracy(df_batch, threshold):

	n_samples = len(df_batch)

	correct = df_batch.correct.sum()

	overall_acc = float(correct)/n_samples

	return overall_acc



def computeMissedDeadlineProb(df_batches, df_inf_time, threshold, t_tar):
	
	missed_deadline = 0
	count = 0
	
	for df_batch, df_batch_inf_time in zip(df_batches, df_inf_time):
		overall_acc = computeOverallAccuracy(df_batch, threshold)
		inference_time = computeInferenceTime(df_batch, df_batch_inf_time, threshold)

		
		missed_deadline += 1 if((overall_acc < threshold) or (inference_time > t_tar)) else 0

		count += 1
		#print(inference_time, inference_time > t_tar, missed_deadline)


	missed_deadline_prob = float(missed_deadline)/count
	#print("Pmd: %s"%missed_deadline_prob)
	return missed_deadline_prob



def computeAvgMissedDeadlineProb(df, df_inf_time, threshold, t_tar, n_rounds, n_batches):

	missed_deadline_rounds = []

	for n_round in range(n_rounds):
		#print("Number of Rounds: %s"%(n_round))

		df = df.sample(frac=1)
		df_batches = chunker(df, batch_size=n_batches)

		df_inf_time = df_inf_time.sample(frac=1)
		df_batches_inf_time = chunker(df_inf_time.inference_time.values, batch_size=n_batches)


		missed_deadline_prob = computeMissedDeadlineProb(df_batches, df_batches_inf_time, threshold, t_tar)

		missed_deadline_rounds.append(missed_deadline_prob)

	missed_deadline_rounds = np.array(missed_deadline_rounds)

	avg_missed_deadline = np.mean(missed_deadline_rounds)

	ic_missed_deadline = compute_confidence_interval(missed_deadline_rounds)

	return avg_missed_deadline, ic_missed_deadline[0], ic_missed_deadline[1] 


def getMissedDeadlineProbThreshold(df, df_inf_time, threshold, t_tar, distortion_lvl_list, n_rounds, n_batches, dist_type_data):
	
	avg_missed_deadline_list, bottom_ic_md_list, upper_ic_md_list = [], [], []

	for distortion_lvl in distortion_lvl_list:
		df_dist_lvl = df[df.distortion_lvl == distortion_lvl]
		df_inf_time_dist_lvl = df_inf_time[df_inf_time.distortion_lvl == distortion_lvl]

		avg_missed_deadline, bottom_ic_md, upper_ic_md = computeAvgMissedDeadlineProb(df_dist_lvl, df_inf_time_dist_lvl, threshold, 
			t_tar, n_rounds, n_batches)

		print("Missed Deadline Probability: %s"%(avg_missed_deadline))

		avg_missed_deadline_list.append(avg_missed_deadline), bottom_ic_md_list.append(bottom_ic_md)
		upper_ic_md_list.append(upper_ic_md)

	result_dict = {"avg_missed_deadline": avg_missed_deadline_list, 
	"bottom_ic_missed_deadline": bottom_ic_md_list, 
	"upper_ic_missed_deadline": upper_ic_md_list, "threshold": len(avg_missed_deadline_list)*[threshold], 
	"t_tar": len(avg_missed_deadline_list)*[t_tar], "distortion_lvl": distortion_lvl_list, 
	"distortion_type_data": len(avg_missed_deadline_list)*[dist_type_data], 
	"n_rounds": len(avg_missed_deadline_list)*[n_rounds], "n_batches": len(avg_missed_deadline_list)*[n_batches],
	"inf_mode": len(avg_missed_deadline_list)*["conventional"]}

	return result_dict



def save_missed_deadline_results(results, savePath):
	df = pd.DataFrame(results)
	df.to_csv(savePath, mode='a', header=not os.path.exists(savePath))



def main(args):

	#missed_deadline_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023", 
	#	"missed_deadline_prob_3_branches_id_%s.csv"%(args.model_id))
	
	#inference_data_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023",  
	#	"inference_data_backbone_id_%s_final_final.csv"%(args.model_id))

	#inference_time_path = os.path.join(config.DIR_NAME, "inference_data_sbrc2023",  
	#	"inference_time_backbone_id_%s_final_final.csv"%(args.model_id))

	missed_deadline_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet", 
		"missed_deadline_prob_backbone_final_final.csv")
	
	inference_data_path = os.path.join(config.DIR_NAME, "inference_data", "caltech256", "mobilenet",  
		"inference_data_backbone_id_%s_final_final.csv"%(args.model_id))

	inference_time_path = os.path.join(config.DIR_NAME, "inference_time_backbone.csv")

	df_inf_data = pd.read_csv(inference_data_path)
	df_inf_time = pd.read_csv(inference_time_path)


	#threshold_list = np.arange(config.threshold_start, config.threshold_end, config.threshold_step)
	#threshold_list = [0.7, 0.8, 0.9]
	threshold_list = [0.8, 0.83]

	t_tar_list = np.arange(config.t_tar_start, config.t_tar_end, config.t_tar_step)

	#df_pristine = df_inf_data[df_inf_data.distortion_type_data == "pristine"]
	df_blur = df_inf_data[df_inf_data.distortion_type_data == "gaussian_blur"]

	distortion_lvl_list = df_inf_time.distortion_lvl.unique()

	for threshold in threshold_list:
		for t_tar in t_tar_list:
			print("Ttar: %s, Threshold: %s"%(t_tar, threshold))

			#pristine_missed_deadline = getMissedDeadlineProbThreshold(df_pristine, df_inf_time, threshold, 
			#	t_tar, args.n_rounds, args.n_batches, dist_type_data="pristine")
			
			blur_missed_deadline = getMissedDeadlineProbThreshold(df_blur, df_inf_time, threshold, 
				t_tar, distortion_lvl_list, args.n_rounds, args.n_batches, dist_type_data="gaussian_blur")
			
			#save_missed_deadline_results(pristine_missed_deadline, missed_deadline_path)
			save_missed_deadline_results(blur_missed_deadline, missed_deadline_path)



if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='UCB using MobileNet')
	parser.add_argument('--model_id', type=int, default=config.model_id, help='Model Id.')
	parser.add_argument('--distortion_type_model', type=str, default=config.distortion_type, help='Distortion Type.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--model_name', type=str, default=config.model_name, help='Model name.')
	parser.add_argument('--cuda', type=bool, default=config.cuda, help='Cuda ?')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--n_rounds', type=float, default=config.n_rounds_outage)
	parser.add_argument('--n_batches', type=int, default=config.batch_size_outage)


	args = parser.parse_args()
	main(args)
