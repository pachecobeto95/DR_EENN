from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config
#from .services import edgeProcessing
#from .services import edgeInference
#import atexit, torch, requests
import torch, requests
#from .services.edgeProcessing import net_config

api = Blueprint("api", __name__, url_prefix="/api")
