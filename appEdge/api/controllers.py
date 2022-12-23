from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
from .services import edgeProcessing
#from .services import edgeInference
#import atexit, torch, requests
import torch, requests
#from .services.edgeProcessing import net_config

api = Blueprint("api", __name__, url_prefix="/api")


@api.route('/edge/edge_ee_inference', methods=["POST"])
def edge_ee_inferece():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	fileImg = request.files['img']
	params = json.load(request.files['data'])

	result = edgeProcessing.eeDnnInference(fileImg, params)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route('/edge/edge_ensemble_inference', methods=["POST"])
def edge_ensemble_inferece():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	fileImg = request.files['img']
	params = json.load(request.files['data'])

	result = edgeProcessing.ensembleDnnInference(fileImg, params)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500



@api.route('/edge/edge_naive_ensemble_inference', methods=["POST"])
def edge_naive_ensemble_inferece():
	"""
	This function receives an image from user or client with smartphone or even a insurance camera 
	into smart sity context
	"""

	fileImg = request.files['img']
	params = json.load(request.files['data'])

	result = edgeProcessing.naiveEnsembleDnnInference(fileImg, params)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
