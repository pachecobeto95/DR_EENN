from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time, torch, cv2, datetime, time, io, utils, json
import numpy as np
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



ee_model = utils.init_ee_dnn(device)

backbone_model = utils.init_backbone_dnn(device)




def eeDnnInference(data_edge):
	#try:

	tensor, conf_list, p_tar, n_branch_edge = torch.Tensor(data_edge["feature"]).to(device), data_edge["conf"], float(data_edge["p_tar"]), int(data_edge["nr_branch_edge"])


	ee_model.eval()

	with torch.no_grad():
		conf_list, infer_class = ee_model.forwardCoEeInferenceCloud(tensor.float(), conf_list, p_tar, n_branch_edge)


	return {"status": "ok"}


def ensembleDnnInference(data_edge):


	tensor, conf_list, p_tar, n_branch_edge = torch.Tensor(data_edge["feature"]).to(device), data_edge["conf"], float(data_edge["p_tar"]), int(data_edge["nr_branch_edge"])

	ee_model.eval()

	with torch.no_grad():
		conf_list, infer_class = ee_model.forwardEnsembleInferenceCloud(tensor.float(), conf_list, p_tar, n_branch_edge)

	return {"status": "ok"}


def naiveEnsembleDnnInference(data_edge):


	tensor, conf_list, p_tar, n_branch_edge = torch.Tensor(data_edge["feature"]).to(device), data_edge["conf"], float(data_edge["p_tar"]), int(data_edge["nr_branch_edge"])

	ee_model.eval()

	with torch.no_grad():
		conf_list, infer_class = ee_model.forwardNaiveEnsembleInferenceCloud(tensor.float(), conf_list, p_tar, n_branch_edge)

	return {"status": "ok"}


def backboneDnnInference(fileImg, params):

	softmax = nn.Softmax(dim=1)
	
	image_bytes = fileImg.read()

	img_tensor = utils.transform_image(image_bytes).to(device)

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	starter.record()

	output = backbone_model(img_tensor)
	conf, inf_class = torch.max(softmax(output), 1)

	ender.record()
	torch.cuda.synchronize()
	inf_time = starter.elapsed_time(ender)

	saveInferenceTime(inf_time, params, device)

	return {"status": "ok"}



def saveInferenceTime(inf_time, params, device):

	result = {"inference_time": [inf_time]}
	result.update(params)

	df = pd.DataFrame(result)
	df.to_csv(config.save_backbone_inf_time_path, mode='a', header=not os.path.exists(config.save_backbone_inf_time_path) )
