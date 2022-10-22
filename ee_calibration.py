import os, time, sys, json, os, argparse, torch, logging
import numpy as np
import pandas as pd
import utils, config, ee_nn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class _ECELoss(nn.Module):
	"""
	Calculates the Expected Calibration Error of a model.
	(This isn't necessary for temperature scaling, just a cool metric).
	The input to this loss is the logits of a model, NOT the softmax scores.
	This divides the confidence outputs into equally-sized interval bins.
	In each bin, we compute the confidence gap:
	bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
	We then return a weighted average of the gaps, based on the number
	of samples in each bin
	See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
	"Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI. 2015.
	"""
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

	def forward(self, logits, labels):
		softmaxes = F.softmax(logits, dim=1)
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)

		ece = torch.zeros(1, device=logits.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

			return ece


class BranchesModelWithTemperature(nn.Module):
	def __init__(self, model, n_branches, device, lr=0.01, max_iter=50):
		super(BranchesModelWithTemperature, self).__init__()
		"""
		This method calibrates a early-exit DNN. The calibration goal is to turn the classification confidencer closer to the real model's accuracy.
		In this work, we apply the calibration method called Temperature Scaling.
		The paper below explains in detail: https://arxiv.org/pdf/1706.04599.pdf

		Here, we follow two approaches:
		* we find a temperature parameter for each side branch
		* we find a temperature parameter for the entire early-exit DNN model.
		"""

		self.model = model			#this receives the architecture model. It is important to notice this models has already trained. 
		self.n_branches = n_branches	#the number of side branches or early exits.
		self.n_exits = self.n_branches + 1 
		self.device = device				 
		self.lr = lr					# defines the learning rate of the calibration process.
		self.max_iter = max_iter		#defines the number of iteractions to train the calibration process

		# This line initiates a parameters list of the temperature 
		self.temperature_branches = [nn.Parameter(1.5*torch.ones(1).to(self.device)) for i in range(self.n_exits)]
		self.softmax = nn.Softmax(dim=1)

		# This line initiates a single temperature parameter for the entire early-exit DNN model
		self.temperature_overall = nn.Parameter(1.5*torch.ones(1).to(self.device))
		self.temperature = nn.Parameter((torch.ones(1) * 1.5).to(self.device))

	def temperature_scale(self, logits):

		temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
		return logits / temperature

	def forwardAllSamplesCalibration(self, x):
		return self.model.forwardAllSamplesCalibration(x, self.temperature_branches)

	def forwardBranchesCalibration(self, x):
		return self.model.forwardBranchesCalibration(x, self.temperature_branches)

	def forwardOverallCalibration(self, x):
		return self.model.forwardOverallCalibration(x, self.temperature)
	
	def temperature_scale_overall(self, logits):
		return torch.div(logits, self.temperature_overall)

	def temperature_scale_branches(self, logits):
		return torch.div(logits, self.temperature_branch)
	
	def save_temperature_branches(self, error_measure_dict, save_branches_path):


		df = pd.read_csv(save_branches_path) if (os.path.exists(save_branches_path)) else pd.DataFrame()

				
		df = df.append(pd.Series(error_measure_dict), ignore_index=True)
		df.to_csv(save_branches_path)

	def save_temperature_overall(self, error_measure_dict, save_overall_path):
		"""
		This method saves the temperature in an csv file in self.save_path
		This saves: 
		p_tar: which means the threshold
		before_temperature_nll: the error before the calibration	
		after_temperature_nll: the error after the calibration
		temperature parameter:
					 
		"""

		df = pd.read_csv(save_overall_path) if (os.path.exists(save_overall_path)) else pd.DataFrame()
		
		df = df.append(pd.Series(error_measure_dict), ignore_index=True)
		df.to_csv(save_overall_path)

	def calibrate_overall2(self, val_loader, p_tar, save_overall_path):
		"""
		This method calibrates the entire model. In other words, this method finds a singles temperature parameter 
		for the entire early-exit DNN model
		"""
	 

		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece = ECE()

		optimizer = optim.LBFGS([self.temperature_overall], lr=self.lr, max_iter=self.max_iter)

		logits_list, labels_list = [], []

		self.model.eval()
		with torch.no_grad():
			for (data, target) in tqdm(val_loader):

				data, target = data.to(self.device), target.to(self.device)
				
				logits, conf, infer_class, exit_branch = self.model(data, p_tar, training=False)

				logits_list.append(logits), labels_list.append(target)

		logits_list = torch.cat(logits_list).to(self.device)
		labels_list = torch.cat(labels_list).to(self.device)

		before_temperature_nll = nll_criterion(logits_list, labels_list).item()
		
		before_ece = ece(logits_list, labels_list).item()

		def eval():
			optimizer.zero_grad()
			loss = nll_criterion(self.temperature_scale_overall(logits_list), labels_list)
			loss.backward()
			return loss
		
		optimizer.step(eval)

		after_temperature_nll = nll_criterion(self.temperature_scale_overall(logits_list), labels_list).item()
		after_ece = ece(self.temperature_scale_overall(logits_list), labels_list).item()

		print("Before NLL: %s, After NLL: %s"%(before_temperature_nll, after_temperature_nll))
		print("Before ECE: %s, After ECE: %s"%(before_ece, after_ece))
		print("Temp %s"%(self.temperature_overall.item()))

		error_measure_dict = {"p_tar": p_tar, "before_nll": before_temperature_nll, "after_nll": after_temperature_nll, 
								"before_ece": before_ece, "after_ece": after_ece, 
								"temperature": self.temperature_overall.item()}
		
		# This saves the parameter to save the temperature parameter
		self.save_temperature_overall(error_measure_dict, save_overall_path)


	def calibrate_overall(self, val_loader, p_tar, save_overall_path):

		self.cuda()
		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece_criterion = _ECELoss().to(self.device)

		# First: collect all the logits and labels for the validation set
		logits_list = []
		labels_list = []
		with torch.no_grad():
			for data, label in tqdm(val_loader):
				data, label = data.to(self.device), label.to(self.device)

				logits, _, _, exit_branch = self.model(data, p_tar, training=False)


				logits_list.append(logits)
				labels_list.append(label)
		
		logits = torch.cat(logits_list).cuda()
		labels = torch.cat(labels_list).cuda()

		# Calculate NLL and ECE before temperature scaling
		before_temperature_nll = nll_criterion(logits, labels).item()
		before_temperature_ece = ece_criterion(logits, labels).item()
		print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

		# Next: optimize the temperature w.r.t. NLL
		optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

		def eval():
			optimizer.zero_grad()
			loss = nll_criterion(self.temperature_scale(logits), labels)
			loss.backward()
			return loss
		
		optimizer.step(eval)

		# Calculate NLL and ECE after temperature scaling
		after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
		after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
		print('Optimal temperature: %.3f' % self.temperature.item())
		print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

		return self

	def calibrate_branches_all_samples(self, val_loader, p_tar, save_branches_path):

		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece = _ECELoss()

		logits_list = [[] for i in range(self.n_exits)]
		labels_list = [[] for i in range(self.n_exits)]

		before_ece_list, after_ece_list = [], []
		
		before_temperature_nll_list, after_temperature_nll_list = [], []

		temperature_branch_list = []

		error_measure_dict = {"p_tar": p_tar}

		self.model.eval()
		with torch.no_grad():
			for (data, target) in tqdm(val_loader):
				
				data, target = data.to(self.device), target.to(self.device)

				logits, _, _ = self.model.forwardAllExits(data)


				for i in range(self.n_exits):
					logits_list[i].append(logits[i])
					labels_list[i].append(target)

		for i in range(self.n_exits):
			print("Exit: %s"%(i))

			if (len(logits_list[i]) == 0):
				before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
				before_ece_list.append(None), after_ece_list.append(None)
				temperature_branch_list.append(None)
				continue

			self.temperature_branch = nn.Parameter((torch.ones(1)*1.0).to(self.device))
			optimizer = optim.LBFGS([self.temperature_branch], lr=self.lr, max_iter=self.max_iter)

			logit_branch = torch.cat(logits_list[i]).to(self.device)
			label_branch = torch.cat(labels_list[i]).to(self.device)

			before_temperature_nll = nll_criterion(logit_branch, label_branch).item()
			before_temperature_nll_list.append(before_temperature_nll)

			before_ece = ece(logit_branch, label_branch).item()
			before_ece_list.append(before_ece)

			def eval():
			optimizer.zero_grad()
			loss = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch)
			loss.backward()
			return loss
			
			optimizer.step(eval)

			after_temperature_nll = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch).item()
			after_temperature_nll_list.append(after_temperature_nll)
			
			after_ece = ece(self.temperature_scale_branches(logit_branch), label_branch).item()
			after_ece_list.append(after_ece)

			print("Branch: %s, Before NLL: %s, After NLL: %s"%(i+1, before_temperature_nll, after_temperature_nll))
			print("Branch: %s, Before ECE: %s, After ECE: %s"%(i+1, before_ece, after_ece))

			print("Temp Branch %s: %s"%(i+1, self.temperature_branch.item()))

			self.temperature_branches[i] = self.temperature_branch

		self.temperature_branches = [temp_branch.item() for temp_branch in self.temperature_branches]
		
		for i in range(self.n_exits):

			error_measure_dict.update({"before_nll_branch_%s"%(i+1): before_temperature_nll_list[i], 
									 "before_ece_branch_%s"%(i+1): before_ece_list[i],
									 "after_nll_branch_%s"%(i+1): after_temperature_nll_list[i],
									 "after_ece_branch_%s"%(i+1): after_ece_list[i],
									 "temperature_branch_%s"%(i+1): self.temperature_branches[i]})

		
		# This saves the parameter to save the temperature parameter for each side branch

		self.save_temperature_branches(error_measure_dict, save_branches_path)

	def calibrate_branches(self, val_loader, dataset, p_tar, save_branches_path, data_augmentation=False):
		"""
		This method calibrates for each side branch. In other words, this method finds a temperature parameter 
		for each side branch of the early-exit DNN model.
		"""

		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece = _ECELoss()
		temperature_branch_list = []

		logits_list = [[] for i in range(self.n_exits)]
		labels_list = [[] for i in range(self.n_exits)]
		idx_sample_exit_list = [[] for i in range(self.n_exits)]
		before_temperature_nll_list, after_temperature_nll_list = [], []
		before_ece_list, after_ece_list = [], []

		error_measure_dict = {"p_tar": p_tar}

		self.model.eval()
		with torch.no_grad():
			for (data, target) in tqdm(val_loader):
				
			data, target = data.to(self.device), target.to(self.device)
			
			logits, _, _, exit_branch = self.model(data, p_tar, training=False)

			logits_list[exit_branch].append(logits)
			labels_list[exit_branch].append(target)


		for i in range(self.n_exits):
			print("Exit: %s"%(i+1))

			if (len(logits_list[i]) == 0):
			before_temperature_nll_list.append(None), after_temperature_nll_list.append(None)
			before_ece_list.append(None), after_ece_list.append(None)
			temperature_branch_list.append(None)
			continue

			self.temperature_branch = nn.Parameter((torch.ones(1)*1.5).to(self.device))
			
			optimizer = optim.LBFGS([self.temperature_branch], lr=self.lr, max_iter=self.max_iter)

			logit_branch = torch.cat(logits_list[i]).to(self.device)
			label_branch = torch.cat(labels_list[i]).to(self.device)

			before_temperature_nll = nll_criterion(logit_branch, label_branch).item()
			before_temperature_nll_list.append(before_temperature_nll)

			before_ece = ece(logit_branch, label_branch).item()
			before_ece_list.append(before_ece)
			weight_list = np.linspace(1, 0.3, self.n_exits)
			def eval():
			optimizer.zero_grad()
			loss = weight_list[i]*nll_criterion(self.temperature_scale_branches(logit_branch), label_branch)
			loss.backward()
			return loss
			
			optimizer.step(eval)

			after_temperature_nll = nll_criterion(self.temperature_scale_branches(logit_branch), label_branch).item()
			after_temperature_nll_list.append(after_temperature_nll)

			after_ece = ece(self.temperature_scale_branches(logit_branch), label_branch).item()
			after_ece_list.append(after_ece)

			
			self.temperature_branches[i] = self.temperature_branch
			#temperature_branch_list.append(self.temperature_branch.item())

			print("Branch: %s, Before NLL: %s, After NLL: %s"%(i+1, before_temperature_nll, after_temperature_nll))
			print("Branch: %s, Before ECE: %s, After ECE: %s"%(i+1, before_ece, after_ece))

			print("Temp Branch %s: %s"%(i+1, self.temperature_branch.item()))

		self.temperature_branches = [temp_branch.item() for temp_branch in self.temperature_branches]

		for i in range(self.n_exits):
			error_measure_dict.update({"before_nll_branch_%s"%(i+1): before_temperature_nll_list[i], 
									 "before_ece_branch_%s"%(i+1): before_ece_list[i],
									 "after_nll_branch_%s"%(i+1): after_temperature_nll_list[i],
									 "after_ece_branch_%s"%(i+1): after_ece_list[i],
									 "temperature_branch_%s"%(i+1): self.temperature_branches[i]})
		
		# This saves the parameter to save the temperature parameter for each side branch

		self.save_temperature_branches(error_measure_dict, save_branches_path)

		return self

def calibrating_early_exit_dnn(model, val_loader, dataset, p_tar, n_branches, device, saveTempBranchesPath):
	print("Calibrating ...")

	overall_calibrated_model = BranchesModelWithTemperature(model, n_branches, device)
	overall_calibrated_model.calibrate_overall(val_loader, p_tar, saveTempBranchesPath["calib_overall"])
	
	branches_calibrated_model = BranchesModelWithTemperature(model, n_branches, device)
	branches_calibrated_model.calibrate_branches(val_loader, dataset, p_tar, saveTempBranchesPath["calib_branches"])


	branches_calibrated_all_samples = BranchesModelWithTemperature(model, n_branches, device)
	branches_calibrated_all_samples.calibrate_branches_all_samples(val_loader, p_tar, 
																	 saveTempBranchesPath["calib_branches_all_samples"])

	calib_models_dict = {"calib_overall": overall_calibrated_model, 
						 "calib_branches": branches_calibrated_model,
						 "calib_all_samples": branches_calibrated_all_samples}
	
	return calib_models_dict

def experiment_early_exit_inference(model, test_loader, p_tar, n_branches, device, model_type="no_calib"):
	df_result = pd.DataFrame()

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list, id_list = [], [], []

	model.eval()

	with torch.no_grad():
	for i, (data, target) in tqdm(enumerate(test_loader, 1)):
		
		data, target = data.to(device), target.to(device)

		if (model_type == "no_calib"):
		_, conf_branches, infered_class_branches = model.forwardAllExits(data)

		elif(model_type == "calib_overall"):
		_, conf_branches, infered_class_branches = model.forwardOverallCalibration(data)

		elif(model_type == "calib_branches"):
		_, conf_branches, infered_class_branches = model.forwardBranchesCalibration(data)

		else:
		_, conf_branches, infered_class_branches = model.forwardAllSamplesCalibration(data)

		conf_branches_list.append([conf.item() for conf in conf_branches])
		infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])	
		correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
		id_list.append(i)
		target_list.append(target.item())

		del data, target
		torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)
	
	print(model_type)
	print("Acc:")
	print([sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)])

	results = {"p_tar": [p_tar]*len(target_list), "target": target_list, "id": id_list}
	for i in range(n_exits):
	results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
					"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
					"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results

def calibrating_early_exit_dnn(model, val_loader, p_tar, n_branches, device, temperatureDict):
	print("Calibrating ...")

	overall_calibrated_model = BranchesModelWithTemperature(model, n_branches, device)
	overall_calibrated_model.calibrate_overall(val_loader, p_tar, temperatureDict["global_calib"])

	branches_calibrated_model = BranchesModelWithTemperature(model, n_branches, device)
	branches_calibrated_model.calibrate_branches(val_loader, dataset, p_tar, temperatureDict["per_branch_calib"])

	branches_calibrated_all_samples = BranchesModelWithTemperature(model, n_branches, device)
	branches_calibrated_all_samples.calibrate_branches_all_samples(val_loader, p_tar, temperatureDict["all_samples_calib"])

	calib_models_dict = {"global_calib": overall_calibrated_model, 
	"per_branch_calib": branches_calibrated_model,
	"all_samples_calib": branches_calibrated_all_samples}

	return calib_models_dict



def run_early_exit_inference(model, test_loader, p_tar, n_branches, device, model_type="no_calib"):
	df_result = pd.DataFrame()

	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, exit_branch_list, id_list = [], [], []

	model.eval()

	with torch.no_grad():
		for (data, target) in tqdm(test_loader):

			data, target = data.to(device), target.to(device)

			if (model_type == "no_calib"):
				_, conf_branches, infered_class_branches = model.forwardInferenceNoCalib(data)

			elif(model_type == "global_calib"):
				_, conf_branches, infered_class_branches = model.forwardGlobalCalibration(data)

			elif(model_type == "per_branch_calib"):
				_, conf_branches, infered_class_branches = model.forwardPerBranchesCalibration(data)
			
			else:
				 _, conf_branches, infered_class_branches = model.forwardAllSamplesCalibration(data)

			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			id_list.append(i)
			target_list.append(target.item())

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)

	results = {"p_tar": [p_tar]*len(target_list), "target": target_list, "id": id_list}
	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results

