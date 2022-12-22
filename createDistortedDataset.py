import os, sys, torchvision
from PIL import Image
from torchvision import datasets, transforms

class DistortionApplier(object):
	def __init__(self, distortion_function, distortion_values):

		self.distortion_function = getattr(self, distortion_function, self.distortion_not_found)
		self.distortion_values = distortion_values

		if(isinstance(distortion_values, list)):
			self.distortion_values = random.choice(distortion_values)

	def __call__(self, img):
				
		return self.distortion_function(img, self.distortion_values)

	def distortion_not_found(self):
		raise Exception("This distortion type has not implemented yet.")

	def gaussian_blur(self, img, distortion_lvl):
		#image = np.array(img)
		#kernel_size = int(4*np.ceil(distortion_lvl/2)+1) if (distortion_lvl < 1) else 	4*distortion_lvl+1
		kernel_size = int(2*np.ceil(distortion_lvl/2)+1) if ( isinstance(distortion_lvl, float) ) else 	2*distortion_lvl+1
		blurrer = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=distortion_lvl)
		return blurrer(img)

	def gaussian_noise(self, img, distortion_lvl):
		
		image = np.array(img)
		noise_img = image + np.random.normal(0, distortion_lvl, (image.shape[0], image.shape[1], image.shape[2]))
		
		return Image.fromarray(np.uint8(noise_img)) 

	def pristine(self, img, distortion_lvl):
		return img

def create_dir(dir_path):

	if(not os.path.exists(dir_path)):
		os.makedirs(dir_path)

def convertDataset(undistorted_dataset_path, target_dataset_path, transform_fc, distortion_type, distortion_lvl):
	dir_list = os.listdir(undistorted_dataset_path)

	for class_dir in dir_list:
		class_dir_path = os.path.join(undistorted_dataset_path, class_dir)
		target_class_dir_path = os.path.join(target_dataset_path, class_dir)
		create_dir(target_class_dir_path)

		file_list = os.listdir(class_dir_path)

		for file_name in file_list:
			# open method used to open different extension image file
			img = Image.open(os.path.join(class_dir_path, file_name))
			img_pil = transform_fc(img)
			save_path = os.path.join(target_class_dir_path, file_name)
			img_pil.save(save_path)


def getInputTranformation(input_dim, dim, distortion_type, distortion_lvl):
	transform = transforms.Compose([transforms.PILToTensor(),
		transforms.Resize((input_dim, input_dim)),
		transforms.CenterCrop((dim, dim)),
		transforms.RandomApply([DistortionApplier(distortion_type, distortion_lvl)], p=1),
		transforms.ToPILImage()])

	return transform


def createDistortedDataset(undistorted_dataset_path, distorted_dataset_path, input_dim, dim, distortion_levels, distortion_type):

	distorted_dataset_path = os.path.join(distorted_dataset_path, distortion_type)
	create_dir(distorted_dataset_path)

	for distortion_lvl in distortion_levels:
		distortion_lvl_dataset_path = os.path.join(distorted_dataset_path, str(distortion_lvl))

		transform_fc = getInputTranformation(input_dim, dim, distortion_type, distortion_lvl)

		create_dir(distortion_lvl_dataset_path)

		convertDataset(undistorted_dataset_path, distortion_lvl_dataset_path, transform_fc, distortion_type, distortion_lvl)


def main(args):

	undistorted_dataset_path = config.dataset_path_dict[args.dataset_name]

	distorted_dataset_path = os.path.join(config.DIR_NAME, "distorted_datasets", "Caltech256")

	n_classes = config.nr_class_dict[args.dataset_name]
	input_dim = config.img_dim_dict[args.n_branches]
	dim = config.dim_dict[args.n_branches]

	distortion_levels = config.distortion_level_dict[distortion_type_data]

	createDistortedDataset(undistorted_dataset_path, distorted_dataset_path, input_dim, dim, distortion_levels, args.distortion_type)

if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description='Create a Distorted Dataset')
	parser.add_argument('--distortion_type', type=str, default="gaussian_blur", help='Distortion Type.')
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, help='Dataset Name.')
	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')
	parser.add_argument('--input_dim', type=int, default=config.input_dim, help='Input Dim. Default: %s'%config.input_dim)
	parser.add_argument('--dim', type=int, default=config.dim, help='Dim. Default: %s'%(config.dim))
	parser.add_argument('--n_branches', type=int, default=3, help='Number of exit exits.')


	args = parser.parse_args()
	main(args)
