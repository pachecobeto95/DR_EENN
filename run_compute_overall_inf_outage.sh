#!/bin/bash

nohup python3 compute_overall_inf_outage_prob.py --n_branches 3 --inf_mode eednn &
nohup python3 compute_overall_inf_outage_prob.py --n_branches 3 --inf_mode ensemble &
nohup python3 compute_conventional_dnn_overall_inf_outage_prob.py --n_branches 3 &
