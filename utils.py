from torchvision import datasets, transforms
import torch, os, sys, requests
import numpy as np
import config
import pandas as pd


def get_indices(dataset, split_ratio):
	
	nr_samples = len(dataset)
	indices = list(range(nr_samples))
	np.random.shuffle(indices)

	train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))


	train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]
	np.random.shuffle(train_val_idx)

	train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

	train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

	return train_idx, val_idx, test_idx


def load_caltech256(args, dataset_path, save_indices_path, seed):
	mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	
	torch.manual_seed(seed)
	np.random.seed(seed=seed)

	transformations_train = transforms.Compose([
		transforms.Resize(args.input_dim),
		transforms.RandomHorizontalFlip(p=args.h_flip_prob),
		transforms.RandomEqualize([args.equalize_prob]),
		transforms.ColorJitter(.4, .4, .4)
		transforms.Resize(args.dim),
		#transforms.CenterCrop(args.dim)
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize(args.input_dim), 
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



