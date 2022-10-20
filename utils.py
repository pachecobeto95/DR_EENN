from torchvision import datasets, transforms
import torch, os, sys, requests, random, logging, torchvision
import numpy as np
import pandas as pd


def get_indices_caltech256(dataset, split_ratio):
	
	nr_samples = len(dataset)
	indices = list(range(nr_samples))
	np.random.shuffle(indices)

	train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))


	train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]
	np.random.shuffle(train_val_idx)

	train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

	train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

	return train_idx, val_idx, test_idx


class DistortionApplier(torchvision.transforms.Lambda):
	def __init__(self, distortion_function, distortion_values):
		super().__init__(distortion_function)
		self.distortion_values = distortion_values

	def __call__(self, img):
		return self.distortion_function(img, self.distortion_values)

def gaussian_blur(img, distortion_values):

	sigma = random.choice(distortion_values)
	kernel_size = (4*sigma+1, 4*sigma+1)
	blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

	return blurrer(img)

def gaussian_noise(img, distortion_values):
	sigma = random.choice(distortion_values) 
	noise_img = img + sigma * torch.randn_like(img)
	return noise_img

def pristine(img, distortion_values):
	return img


def motion_blur(self, img, distortion_values):

	img = np.array(img)
	sigma = random.choice(distortion_values)

	# generating the kernel
	kernel_motion_blur = np.zeros((sigma, sigma))
	kernel_motion_blur[int((sigma-1)/2), :] = np.ones(sigma)
	kernel_motion_blur = kernel_motion_blur / sigma

	# applying the kernel to the input image
	blurred_img = cv2.filter2D(img, -1, kernel_motion_blur)

	return blurred_img




class DistortionApplier2(object):
	def __init__(self, distortion_function, distortion_values):

		self.distortion_values = distortion_values
		self.distortion_function = getattr(self, distortion_function, self.distortion_not_found)

	def __call__(self, img):
		return self.distortion_function(img, self.distortion_values)

	def gaussian_blur(self, img, distortion_values):
		image = np.array(img)
		sigma = random.choice(distortion_values)
		blurred_img = cv2.GaussianBlur(image, (4*sigma+1, 4*sigma+1), sigma, None, sigma, cv2.BORDER_CONSTANT)
		return Image.fromarray(blurred_img) 

	def gaussian_noise(self, img, distortion_values):
		image = np.array(img)
		sigma = random.choice(distortion_values)
		noise_img = image + np.random.normal(0, sigma, (image.shape[0], image.shape[1], image.shape[2]))
		return Image.fromarray(np.uint8(noise_img)) 

	def pristine(self, img, distortion_values):
		return img

	def motion_blur(self, img, distortion_values):

		img = np.array(img)
		sigma = random.choice(distortion_values)

		# generating the kernel
		kernel_motion_blur = np.zeros((sigma, sigma))
		kernel_motion_blur[int((sigma-1)/2), :] = np.ones(sigma)
		kernel_motion_blur = kernel_motion_blur / sigma

		# applying the kernel to the input image
		blurred_img = cv2.filter2D(img, -1, kernel_motion_blur)

		return blurred_img

	def distortion_not_found(self):
		raise Exception("This distortion type has not implemented yet.")

def load_caltech256(args, dataset_path, save_indices_path, distortion_values):
	mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	print(dataset_path)
	torch.manual_seed(args.seed)
	np.random.seed(seed=args.seed)

	transformations_train = transforms.Compose([
		#transforms.Resize(args.input_dim),
		#transforms.RandomHorizontalFlip(p=args.h_flip_prob),
		#transforms.RandomEqualize([args.equalize_prob]),
		#transforms.ColorJitter(.4, .4, .4),
		#transforms.RandomApply([DistortionApplier2(args.distortion_type, distortion_values)], p=0.5),
		transforms.Resize(args.dim),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.RandomApply([DistortionApplier2(args.distortion_type, distortion_values)], p=0.5),
		transforms.Resize(args.dim), 
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)


	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	train_idx_path = os.path.join(save_indices_path, "training_idx_caltech256.npy")
	val_idx_path = os.path.join(save_indices_path, "validation_idx_caltech256.npy")
	test_idx_path = os.path.join(save_indices_path, "test_idx_caltech256.npy")


	if( os.path.exists(train_idx_path) ):
		#Load the indices to always use the same indices for training, validating and testing.
		train_idx = np.load(train_idx_path)
		val_idx = np.load(val_idx_path)
		test_idx = np.load(test_idx_path)

	else:
		# This line get the indices of the samples which belong to the training dataset and test dataset. 
		train_idx, val_idx, test_idx = get_indices_caltech256(train_set, args.split_ratio)

		#Save the training, validation and testing indices.
		np.save(train_idx_path, train_idx), np.save(val_idx_path, val_idx), np.save(test_idx_path, test_idx)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

	return train_loader, val_loader, test_loader

def compute_metrics(criterion, output_list, conf_list, class_list, target, loss_weights):
	model_loss = 0
	ee_loss = []

	for output, inf_class, weight in zip(output_list, class_list, loss_weights):
		loss_branch = criterion(output, target)
		model_loss += weight*loss_branch

		acc_branch = 100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0)

		ee_loss.append(loss_branch.item()), acc_branches.append(acc_branch)

	acc_model = np.mean(np.array(acc_branches))

	return model_loss, ee_loss, acc_model, acc_branches

def trainEEDNNs(model, train_loader, optimizer, criterion, n_exits, epoch, device, loss_weights):

	model_loss_list, ee_loss_list = [], []
	model_acc_list, ee_acc_list = [], []
	#train_acc_dict = {i: [] for i in range(1, n_exits+1)}

	model.train()
	print(train_loader)
	for i, (data, target) in enumerate(train_loader, 1):
		data, target = data.to(device), target.to(device)

		output_list, conf_list, class_list = model.forwardTrain(data)
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

	print("Epoch: %s, Model Loss: %s, Model Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "train_loss": avg_loss, "train_acc": avg_acc}

	for i in range(n_exits):
		result_dict.update({"train_ee_acc_%s"%(i+1): avg_ee_acc[i], "train_ee_loss_%s"%(i+1): avg_ee_loss[i]})
		print("Train EE Loss: %s, Train EE Acc: %s"%(avg_ee_acc[i], avg_ee_loss[i]))

	return result_dict

def evalEEDNNs(model, val_loader, criterion, n_exits, epoch, device, loss_weights):
	model_loss_list, ee_loss_list = [], []
	model_acc_list, ee_acc_list = [], []

	model.eval()

	with torch.no_grad():
		for i, (data, target) in tqdm(enumerate(val_loader, 1)):
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

	print("Epoch: %s, Model Loss: %s, Model Acc: %s"%(epoch, avg_loss, avg_acc))

	result_dict = {"epoch": epoch, "val_loss": avg_loss, "val_acc": avg_acc}

	for i in range(n_exits):
		result_dict.update({"val_ee_acc_%s"%(i+1): avg_ee_acc[i], "val_ee_loss_%s"%(i+1): avg_ee_loss[i]})
		print("Val EE Loss: %s, Val EE Acc: %s"%(avg_ee_acc[i], avg_ee_loss[i]))

	return result_dict

