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

backbone_model = utils.init_backbone_dnn(device)


def backboneDnnInference(data):

	softmax = nn.Softmax(dim=1)
	
	img_tensor = torch.Tensor(data["img"]).to(device)

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
