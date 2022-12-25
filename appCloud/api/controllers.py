from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
from .services import cloudProcessing, cloudProcessing_alt

api = Blueprint("api", __name__, url_prefix="/api")


# Define url for the user send the image
@api.route('/cloud/cloud_ee_inference', methods=["POST"])
def cloud_ee_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	"""

	data_edge = request.json

	result = cloudProcessing.eeDnnInference(data_edge)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500



@api.route('/cloud/cloud_ensemble_inference', methods=["POST"])
def cloud_ensemble_inferece():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	data_edge = request.json

	result = cloudProcessing.ensembleDnnInference(data_edge)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500



@api.route('/cloud/cloud_naive_ensemble_inference', methods=["POST"])
def cloud_naive_ensemble_inferece():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	data_edge = request.json

	result = cloudProcessing.naiveEnsembleDnnInference(data_edge)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500



@api.route('/cloud/cloud_backbone_inference', methods=["POST"])
def cloud_backbone_inference():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""


	fileImg = request.files['img']
	params = json.load(request.files['data'])

	result = cloudProcessing.backboneDnnInference(fileImg, params)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500




@api.route('/cloud/cloud_backbone_inference_alt', methods=["POST"])
def cloud_backbone_inference_alt():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	data = request.json

	result = cloudProcessing_alt.backboneDnnInference(data)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500













