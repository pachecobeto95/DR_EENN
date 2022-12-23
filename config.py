import os

DIR_NAME = os.path.dirname(__file__)
DEBUG = True


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
timeout = 10

nr_class_dict = {"caltech256": 258}
blur_levels = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2]
#distortion_level_dict = {"pristine": [0], "gaussian_blur": [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.2, 2.5, 2.7, 2.8, 3], 
#"gaussian_noise": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

distortion_level_dict = {"pristine": [0], "gaussian_blur": blur_levels, 
"gaussian_noise": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


distortion_type_list = ["pristine", "gaussian_blur", "gaussian_noise"]

threshold_list = [0.7, 0.75, 0.8, 0.85, 0.9]

img_dim_dict = {1: 256, 3: 330, 5: 256}

dim_dict = {1: 224, 3: 300, 5: 224}


logFile = os.path.join(DIR_NAME, "logging_training_%s_%s_%s.csv"%(dataset_name, model_name, model_id))

dataset_path_dict = {"caltech256": os.path.join(DIR_NAME, "datasets", "caltech256", "257_ObjectCategories")}

distorted_dataset_path = os.path.join(DIR_NAME, "distorted_datasets", "Caltech256", "gaussian_blur")




indices_path_dict = {"caltech256": os.path.join(DIR_NAME, "indices", "caltech256")}


#x_axis_str_dict = {"gaussian_blur": r"Gaussian Blur ($\sigma_{B}$)", "gaussian_noise": r"Gaussian Noise ($\sigma_{N}$)"}
x_axis_str_dict = {"gaussian_blur": r"Blur Gaussiano ($\sigma_{B}$)", "gaussian_noise": r"Ruído Gaussiano ($\sigma_{N}$)"}


fontsize = 18
color_list = ["black", "blue", "red", "magenta"]
line_style_list = ["dotted", "dashed", "solid", "dashdot"]
marker_list = ["^", "v", ".", "<"]
shouldSave = True

plot_dict = {"fontsize": fontsize, 
"color": color_list, 
"line_style": line_style_list,
"marker": marker_list,
"x_axis": x_axis_str_dict, 
"shouldSave": shouldSave}


save_inf_time_path = os.path.join(DIR_NAME, "inference_time.csv")
save_backbone_inf_time_path = os.path.join(DIR_NAME, "backbone_inference_time.csv")

# Edge URL Configuration 
HOST_EDGE = "146.164.69.165"
PORT_EDGE = 5001
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)

# Cloud URL Configuration 
HOST_CLOUD = "146.164.69.144"
PORT_CLOUD = 3001
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)


url_ee = "%s/api/edge/edge_ee_inference"%(URL_EDGE)
url_ensemble = "%s/api/edge/edge_ensemble_inference"%(URL_EDGE)
url_cloud = "%s/api/cloud/cloud_backbone_inference"%(URL_CLOUD)



ee_model_path = os.path.join(DIR_NAME, "models", "caltech256", "mobilenet", "pristine_ee_model_mobilenet_3_branches_id_1.pth")
backbone_dnn_path = os.path.join(DIR_NAME, "models", "caltech256", "mobilenet", "pristine_backbone_model_mobilenet_1.pth")


ee_inference_data_path = os.path.join(DIR_NAME, "inference_data", "caltech256", "mobilenet", "inference_data_3_branches_id_1_final_final.csv")

