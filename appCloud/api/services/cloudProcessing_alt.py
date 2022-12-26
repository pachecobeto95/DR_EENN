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


def backboneDnnInference(img, params):

	softmax = nn.Softmax(dim=1)
	
	img_tensor = torch.Tensor(img).to(device)


	output = backbone_model(img_tensor)
	conf, inf_class = torch.max(softmax(output), 1)

	return {"status": "ok"}
