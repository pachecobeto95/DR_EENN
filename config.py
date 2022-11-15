import os

DIR_NAME = os.path.dirname(__file__)

dataset_name = "caltech256"
model_name = "mobilenet"
split_ratio = 0.1
batch_size_train = 64
h_flip_prob = 0.5
equalize_prob = 0.5
input_dim = 256
dim = 224
seed = 42
cuda = True
n_branches = 1
exit_type = "bnpool"
pretrained = True
distortion_type = "pristine"
epochs = 200
max_patience = 20
model_id = 1
distribution = "linear"

nr_class_dict = {"caltech256": 258}
distortion_level_dict = {"pristine": [0], "gaussian_blur": [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5, 2.7, 2.8, 3], 
"gaussian_noise": [1, 2, 3, 5, 10, 12, 13, 15, 18, 20, 22, 25]}

img_dim_dict = {1: 256, 3: 330, 5: 256}

dim_dict = {1: 224, 3: 300, 5: 224}


logFile = os.path.join(DIR_NAME, "logging_training_%s_%s_%s.csv"%(dataset_name, model_name, model_id))

dataset_path_dict = {"caltech256": os.path.join(DIR_NAME, "datasets", "caltech256", "257_ObjectCategories")}

indices_path_dict = {"caltech256": os.path.join(DIR_NAME, "indices", "caltech256")}

