from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time, utils, torch, datetime, io, json
import numpy as np
import torchvision.models as models
import pandas as pd
#from .utils import loadDistortionClassifier, inferenceTransformationDistortionClassifier
#from .utils import init_b_mobilenet, select_distorted_model, inferenceTransformation, read_temperature, BranchesModelWithTemperature
import torchvision.transforms as transforms
from PIL import Image
#from .mobilenet import B_MobileNet
#from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eeDnnInference(fileImg, params):


	tensor = utils.transform_image(fileImg).to(device)

	return {"status": "ok"}
	"""
	img_np = np.array(Image.open(io.BytesIO(image_bytes)))
	response_request = {"status": "ok"}

	dist_prop = "robust" if robust else "standard"
	start = time.time()
	if (robust):
		distortion_type = distortionClassifierInference(img_np, distortion_classifier).item()
	else:
		distortion_type = -1

	#distortion_type = distortionClassifierInference(img_np, distortion_classifier)
	output, conf_list, infer_class = b_mobileNetInferenceEdge(tensor, distorted_model_list, distorted_temp_list, distortion_type, device)

	if (infer_class is None):
		response_request = sendToCloud(config.url_cloud_ee, output, conf_list, distortion_type)
	inference_time = time.time() - start
	if (response_request["status"]=="ok"):
		saveInferenceTime(inference_time, distortion_classes[distortion_type.item()], end_dist_type, distortion_lvl, dist_prop)

	return response_request
	"""