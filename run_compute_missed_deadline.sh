#!/bin/bash

nohup python3 compute_missed_deadline_eednn_based.py --n_branches 3 --inf_mode eednn &
nohup python3 compute_missed_deadline_eednn_based.py --n_branches 3 --inf_mode ensemble &
nohup python3 compute_missed_deadline_conventional_dnn.py