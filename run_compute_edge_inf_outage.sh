#!/bin/bash

nohup python3 compute_edge_inf_outage_prob.py --n_branches 3 --inf_mode eednn &
nohup python3 compute_edge_inf_outage_prob.py --n_branches 3 --inf_mode ensemble &
