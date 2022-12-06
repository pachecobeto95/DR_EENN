#!/bin/bash

python3 extracting_inference_data.py --model_id 1 --n_branches 3 --distortion_type pristine
#python3 extracting_inference_data.py --model_id 1 --n_branches 3 --distortion_type gaussian_blur
#python3 extracting_inference_data.py --model_id 1 --n_branches 3 --distortion_type gaussian_noise
