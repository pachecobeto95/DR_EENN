from torchvision import datasets, transforms
import torch, os, sys, requests, random, logging, torchvision, config, ee_nn, b_mobilenet
import numpy as np
import pandas as pd
from PIL import Image

def get_indices(dataset, split_ratio):
	
	nr_samples = len(dataset)
	indices = list(range(nr_samples))
	np.random.shuffle(indices)

	train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))


	train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]

	np.random.shuffle(train_val_idx)

	train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

	train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

	return train_val_idx, test_idx, test_idx


class DistortionApplier(object):
	def __init__(self, distortion_function, distortion_values):

		self.distortion_function = getattr(self, distortion_function, self.distortion_not_found)
		self.distortion_values = distortion_values

		if(isinstance(distortion_values, list)):
			self.distortion_values = random.choice(distortion_values)

	def __call__(self, img):
				
		return self.distortion_function(img, self.distortion_values)

	def gaussian_blur(self, img, distortion_lvl):
		#image = np.array(img)
		kernel_size = int(4*np.ceil(distortion_lvl/2)+1) if (distortion_lvl < 1) else 	4*distortion_lvl+1
		blurrer = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=distortion_lvl)
		return blurrer(img)

	def gaussian_noise(self, img, distortion_lvl):
		
		image = np.array(img)
		noise_img = image + np.random.normal(0, distortion_lvl, (image.shape[0], image.shape[1], image.shape[2]))
		
		return Image.fromarray(np.uint8(noise_img)) 

	def pristine(self, img, distortion_lvl):
		return img

	def motion_blur(self, img, distortion_lvl):

		img = np.array(img)

		# generating the kernel
		kernel_motion_blur = np.zeros((distortion_lvl, distortion_lvl))
		kernel_motion_blur[int((distortion_lvl-1)/2), :] = np.ones(distortion_lvl)
		kernel_motion_blur = kernel_motion_blur / distortion_lvl

		# applying the kernel to the input image
		blurred_img = cv2.filter2D(img, -1, kernel_motion_blur)
		return blurred_img

	def distortion_not_found(self):
		raise Exception("This distortion type has not implemented yet.")


def load_caltech256(args, dataset_path, save_indices_path, input_dim, dim, distortion_type, distortion_values):
	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)

	transformations_train = transforms.Compose([
		transforms.Resize((input_dim, input_dim)),
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.CenterCrop((dim, dim)),
		transforms.RandomHorizontalFlip(p=0.25),
		transforms.RandomRotation(25),
		transforms.RandomApply([DistortionApplier(distortion_type, distortion_values)], p=args.distortion_prob),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((input_dim, input_dim)),
		transforms.CenterCrop((dim, dim)),
		transforms.RandomApply([DistortionApplier(distortion_type, distortion_values)], p=args.distortion_prob),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)


	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	train_idx_path = os.path.join(save_indices_path, "training_idx_caltech256_%s.npy"%(args.model_id))
	val_idx_path = os.path.join(save_indices_path, "validation_idx_caltech256_%s.npy"%(args.model_id))
	test_idx_path = os.path.join(save_indices_path, "test_idx_caltech256.npy")

	if( os.path.exists(train_idx_path) ):
		#Load the indices to always use the same indices for training, validating and testing.
		train_idx = np.load(train_idx_path)
		val_idx = np.load(val_idx_path)
		test_idx = np.load(test_idx_path)

	else:
		# This line get the indices of the samples which belong to the training dataset and test dataset. 
		train_val_idx, test_idx = get_indices(train_set, args.split_ratio)
		train_idx, val_idx = get_indices(train_set, args.split_ratio)

		#Save the training, validation and testing indices.
		np.save(train_idx_path, train_idx), np.save(val_idx_path, val_idx), np.save(test_idx_path, test_idx)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

	return train_loader, val_loader, val_loader


def compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights):
	model_loss = 0
	ee_loss, acc_branches = [], []

	for i, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
		loss_branch = criterion(output, target)
		model_loss += weight*loss_branch

		acc_branch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)

		ee_loss.append(loss_branch.item()), acc_branches.append(acc_branch)

	acc_model = np.mean(np.array(acc_branches))

	return model_loss, ee_loss, acc_model, acc_branches

def trainBackboneDNN(model, train_loader, optimizer, criterion, epoch, device):

	loss_list, acc_list = []

	#logging.basicConfig(level=logging.DEBUG, filename=config.logFile, filemode="a+")
	model.train()

	for (data, target) in tqdm(train_loader):
		data, target = data.to(device), target.to(device)

		output = model(data)
		_, infered_class = torch.max(self.softmax(output_branch), 1)

		optimizer.zero_grad()

		loss = criterion(output, target)

		acc_batch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)

		loss_list.append(loss.item()), acc_list.append(acc_batch)

		model_loss.backward()
		optimizer.step()

		# clear variables
		del data, target, output, infered_class
		torch.cuda.empty_cache()


	avg_loss = round(np.mean(loss_list), 4)

	avg_acc = round(np.mean(acc_list), 2)

	print("Epoch: %s, Train Loss: %s, Train Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "train_loss": avg_loss, "train_acc": avg_acc}

	return result_dict


def trainEEDNNs(model, train_loader, optimizer, criterion, n_exits, epoch, device, loss_weights):

	model_loss_list, ee_loss_list = [], []
	model_acc_list, ee_acc_list = [], []
	#train_acc_dict = {i: [] for i in range(1, n_exits+1)}

	#logging.basicConfig(level=logging.DEBUG, filename=config.logFile, filemode="a+")
	model.train()

	for (data, target) in tqdm(train_loader):
		data, target = data.to(device), target.to(device)

		output_list, conf_list, class_list = model(data)
		optimizer.zero_grad()

		model_loss, ee_loss, model_acc, ee_acc = compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights)

		model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss)

		model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)

		model_loss.backward()
		optimizer.step()

		# clear variables
		del data, target, output_list, conf_list, class_list
		torch.cuda.empty_cache()


	avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)

	avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

	#logging.debug("Epoch: %s, Train Model Loss: %s, Train Model Acc: %s"%(epoch, avg_loss, avg_acc))
	print("Epoch: %s, Train Model Loss: %s, Train Model Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "train_loss": avg_loss, "train_acc": avg_acc}

	for i in range(n_exits):
		result_dict.update({"train_ee_acc_%s"%(i+1): avg_ee_acc[i], "train_ee_loss_%s"%(i+1): avg_ee_loss[i]})
		#logging.debug("Epoch: %s, Train Loss EE %s: %s, Train Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))
		print("Epoch: %s, Train Loss EE %s: %s, Train Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))
	return result_dict


def evalBackboneDNN(model, train_loader, optimizer, criterion, epoch, device):

	loss_list, acc_list = [], []

	model.eval()
	#logging.basicConfig(level=logging.DEBUG, filename=config.logFile, filemode="a+")

	with torch.no_grad():
		for (data, target) in tqdm(val_loader):
			data, target = data.to(device), target.to(device)

			output = model(data)

			_, infered_class = torch.max(self.softmax(output_branch), 1)

			optimizer.zero_grad()

			loss = criterion(output, target)

			acc_batch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)

			loss_list.append(loss.item()), acc_list.append(acc_batch)

			# clear variables
			del data, target, output, infered_class
			torch.cuda.empty_cache()

	avg_loss = round(np.mean(loss_list), 4)

	avg_acc = round(np.mean(acc_list), 2)

	#logging.debug("Epoch: %s, Val Model Loss: %s, Val Model Acc: %s"%(epoch, avg_loss, avg_acc))
	print("Epoch: %s, Val Loss: %s, Val Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "val_loss": avg_loss, "val_acc": avg_acc}

	return result_dict


def evalEEDNNs(model, val_loader, criterion, n_exits, epoch, device, loss_weights):
	model_loss_list, ee_loss_list = [], []
	model_acc_list, ee_acc_list = [], []

	model.eval()
	#logging.basicConfig(level=logging.DEBUG, filename=config.logFile, filemode="a+")

	with torch.no_grad():
		for (data, target) in tqdm(val_loader):
			data, target = data.to(device), target.to(device)

			output_list, conf_list, class_list = model.forwardTrain(data)

			model_loss, ee_loss, model_acc, ee_acc = compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights)

			model_loss_list.append(float(model_loss.item())), ee_loss_list.append(ee_loss)
			model_acc_list.append(model_acc), ee_acc_list.append(ee_acc)


			# clear variables
			del data, target, output_list, conf_list, class_list
			torch.cuda.empty_cache()

	avg_loss, avg_ee_loss = round(np.mean(model_loss_list), 4), np.mean(ee_loss_list, axis=0)

	avg_acc, avg_ee_acc = round(np.mean(model_acc_list), 2), np.mean(ee_acc_list, axis=0)

	#logging.debug("Epoch: %s, Val Model Loss: %s, Val Model Acc: %s"%(epoch, avg_loss, avg_acc))
	print("Epoch: %s, Val Model Loss: %s, Val Model Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "val_loss": avg_loss, "val_acc": avg_acc}

	for i in range(n_exits):
		result_dict.update({"val_ee_acc_%s"%(i+1): avg_ee_acc[i], "val_ee_loss_%s"%(i+1): avg_ee_loss[i]})
		#logging.debug("Epoch: %s, Val Loss EE %s: %s, Val Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))
		print("Epoch: %s, Val Loss EE %s: %s, Val Acc EE %s: %s"%(epoch, i, avg_ee_loss[i], i, avg_ee_acc[i]))
	return result_dict


def load_ee_dnn(args, model_path, n_classes, dim, device):
	#Load the trained early-exit DNN model.

	if (args.n_branches == 1):

		ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.dim, device, args.exit_type, 
			args.distribution)

	elif(args.n_branches == 3):
		ee_model = b_mobilenet.B_MobileNet(n_classes, args.pretrained, args.n_branches, dim, args.exit_type, device)

	elif(args.n_branches == 5):

		ee_model = ee_nn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, args.dim, device, args.exit_type, 
			args.distribution)

	else:
		raise Exception("The number of early-exit branches is not available yet.")


	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

	ee_model = ee_model.to(device)
	ee_model.eval()

	return ee_model