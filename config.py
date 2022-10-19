import os

DIR_NAME = os.path.dirname(__file__)

dataset_name = "caltech256"
model_name = "mobilenet"
split_ratio = 0.1
batch_size_train = 256
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

nr_class_dict = {"caltech256": 256}
distortion_level_dict = {"pristine": [0], "gaussian_blur": [1, 2, 3, 4, 5], "gaussian_noise": [5, 10, 20, 30, 40]}

