from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time, utils, torch, datetime, io, json
import numpy as np
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from .b_mobilenet import B_MobileNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ee_model = utils.init_ee_dnn(device)


df_ee = pd.read_csv(config.ee_inference_data_path)
df_ee = df_ee.loc[:, ~df_ee.columns.str.contains('^Unnamed')]



def extractData(df, distortion_lvl, distortion_type_data):

	df = df[df.distortion_type_data=="pristine"] if (distortion_lvl==0) else df[df.distortion_type_data==distortion_type_data]

	df = df[df.distortion_lvl == distortion_lvl]

	return df


def compute_acc_branches(df_ee, distortion_lvl, distortion_type):

	acc_list = []
	n_branches = 3
	n_exits = n_branches + 1

	df = extractData(df_ee, distortion_lvl, distortion_type)

	for i in range(n_exits):

		acc_branch = sum(df["correct_branch_%s"%(i+1)])/len(df["correct_branch_%s"%(i+1)])
		acc_list.append(acc_branch)

	return acc_list


def eeDnnInference(fileImg, params):

	response_request = {"status": "ok"}

	image_bytes = fileImg.read()

	img_tensor = utils.transform_image(image_bytes).to(device)

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	starter.record()


	output, conf_list, infer_class, wasClassified = run_ee_dnn_inference(img_tensor, params["distortion_type"], int(params["nr_branch_edge"]), float(params["p_tar"]), device)


	if (not wasClassified):
		response_request = sendToCloud(config.url_cloud_ee, output, conf_list, params)

	ender.record()
	torch.cuda.synchronize()
	inf_time = starter.elapsed_time(ender)

	if (response_request["status"]=="ok"):
		saveInferenceTime(inf_time, params, device)

	return response_request


def ensembleDnnInference(fileImg, params):

	response_request = {"status": "ok"}

	image_bytes = fileImg.read()

	img_tensor = utils.transform_image(image_bytes).to(device)


	acc_branches = compute_acc_branches(df_ee, float(params["distortion_lvl"]), params["distortion_type"])

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	starter.record()

	output, conf_list, infer_class, wasClassified = run_ensemble_dnn_inference(img_tensor, params, acc_branches, device)


	if (not wasClassified):
		response_request = sendToCloud(config.url_cloud_ensemble, output, conf_list, params)

	ender.record()
	torch.cuda.synchronize()
	inf_time = starter.elapsed_time(ender)

	if (response_request["status"]=="ok"):
		saveInferenceTime(inf_time, params, device)

	return response_request

def naiveEnsembleDnnInference(fileImg, params):

	response_request = {"status": "ok"}

	image_bytes = fileImg.read()

	img_tensor = utils.transform_image(image_bytes).to(device)

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	starter.record()

	output, conf_list, infer_class, wasClassified = run_naive_ensemble_dnn_inference(img_tensor, params["distortion_type"], int(params["nr_branch_edge"]), float(params["p_tar"]), device)

	return {"status": "ok"}


	if (not wasClassified):
		response_request = sendToCloud(config.url_cloud_naive_ensemble, output, conf_list, params)

	ender.record()
	torch.cuda.synchronize()
	inf_time = starter.elapsed_time(ender)

	if (response_request["status"]=="ok"):
		saveInferenceTime(inf_time, params, device)

	return response_request


def run_ee_dnn_inference(img_tensor, distortion_type, nr_branch_edge, p_tar, device):

	ee_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class, wasClassified = ee_model.forwardEeInference(img_tensor.float(), nr_branch_edge, p_tar)
	
	return output, conf_list, infer_class, wasClassified


def run_ensemble_dnn_inference(img_tensor, params, acc_branches, device):


	p_tar, n_branch_edge =  float(params["p_tar"]), int(params["nr_branch_edge"])


	ee_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class, wasClassified = ee_model.forwardEnsembleInference(img_tensor.float(), acc_branches, n_branch_edge, p_tar, device)
	
	return output, conf_list, infer_class, wasClassified


def run_naive_ensemble_dnn_inference(img_tensor, distortion_type, nr_branch_edge, p_tar, device):

	ee_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class, wasClassified = ee_model.forwardNaiveEnsembleInference(img_tensor.float(), nr_branch_edge, p_tar)
	
	return output, conf_list, infer_class, wasClassified


def saveInferenceTime(inf_time, params, device):

	result = {"inference_time": inf_time}
	result.update(params)

	df = pd.DataFrame(result)
	df.to_csv(config.save_inf_time_path, mode='a', header=not os.path.exists(config.save_inf_time_path) )


def sendToCloud(url, feature_map, conf_list, params):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	partitioning_layer (int): partitioning layer decided by the optimization method. 
	"""

	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": conf_list}
	data.update(params)

	try:
		r = requests.post(url, json=data, timeout=config.timeout)
	except requests.exceptions.ConnectTimeout:
		return {"status": "error"}

	if (r.status_code != 200 and r.status_code != 201):
		return {"status": "error"}

	else:
		return {"status": "ok"}
