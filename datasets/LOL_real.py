from torch.utils.data import Dataset
from PIL import Image
# from utils import data_utils
import os
import glob
import numpy as np

import torch
import random
import cv2



def singleScaleRetinex(img,sigma):
	temp=cv2.GaussianBlur(img,(0,0),sigma)
	gaussian=np.where(temp==0,0.01,temp)
	retinex=np.log10(img+0.01)-np.log10(gaussian)
	return retinex

def multiScaleRetinex(img,sigma_list):
	retinex = np.zeros_like(img*1.0)
	for sigma in sigma_list:
		retinex+=singleScaleRetinex(img,sigma)
	retinex = retinex/len(sigma_list)
	return retinex

def colorRestoration(img,alpha,beta):
	img_sum=np.sum(img,axis=2,keepdims=True)
	color_restoration = beta*(np.log10(alpha*img)-np.log10(img_sum))
	return color_restoration

def simplestColorBalance(img,low_clip,high_clip):
	total=img.shape[0]*img.shape[1]
	for i in range(img.shape[2]):
		unique,counts=np.unique(img[:,:,i],return_counts=True)
		current=0
		for u,c in zip(unique,counts):
			if float(current)/total<low_clip:
				low_val=u
			if float(current)/total<high_clip:
				high_val=u
			current+=c
		img[:,:,i]=np.maximum(np.minimum(img[:,:,i],high_val),low_val)

	return img

def MSRCR(img,sigma_list=[15,80,250],G=5,b=25,alpha=125,beta=46,low_clip=0.01,high_clip=0.99):
	img = np.float64(img)+1.0

	img_retinex=multiScaleRetinex(img,sigma_list)
	img_color=colorRestoration(img,alpha,beta)
	img_msrcr=G*(img_retinex*img_color+b)
	for i in range(img_msrcr.shape[2]):
		img_msrcr[:,:,i]=(img_msrcr[:,:,i]-np.min(img_msrcr[:,:,i]))/(np.max(img_msrcr[:,:,i])-np.min(img_msrcr[:,:,i]))*255

	img_msrcr=np.uint8(np.minimum(np.maximum(img_msrcr,0),255))
	img_msrcr=simplestColorBalance(img_msrcr,low_clip,high_clip)
	return img_msrcr



def glob_file_list(root):
	return sorted(glob.glob(os.path.join(root, '*')))


def flip(x, dim):
	indices = [slice(None)] * x.dim()
	indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
	return x[tuple(indices)]



def augment_torch(img_list, hflip=True, rot=True):
	hflip = hflip and random.random() < 0.5
	vflip = rot and random.random() < 0.5
	def _augment(img):
		if hflip:
			img = flip(img, 2)
		if vflip:
			img = flip(img, 1)
		return img
	return [_augment(img) for img in img_list]


class ImagesDataset2(Dataset):
	def __init__(self, source_root_pre, target_root_pre, opts, target_transform=None, source_transform=None, train=1):
		self.source_transform = source_transform
		self.target_transform = target_transform
		#数据集路径
		if train:
			source_root = '/mnt/jxsd_jaw/motongstudio/zx/data/Train/low/'
			target_root = '/mnt/jxsd_jaw/motongstudio/zx/data/Train/high/'
		else:
			source_root = '/mnt/jxsd_jaw/motongstudio/zx/data/Test/low/'
			target_root = '/mnt/jxsd_jaw/motongstudio/zx/data/Test/high/'


		script_dir = os.path.dirname(os.path.abspath(__file__))
		# source_root = os.path.join(source_root_pre, source_root)
		# target_root = os.path.join(target_root_pre, target_root)
		subfolders_LQ = glob_file_list(source_root)
		subfolders_GT = glob_file_list(target_root)


		self.source_paths = []
		self.target_paths = []
		self.train = train

		for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
			subfolder_name = os.path.basename(subfolder_GT)

			img_paths_LQ = [subfolder_LQ]
			img_paths_GT = [subfolder_GT]

			max_idx = len(img_paths_LQ)
			assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
			self.source_paths.extend(img_paths_LQ)
			self.target_paths.extend(img_paths_GT)

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = cv2.imread(from_path)
		#物理先验知识
		# from_im_light =MSRCR(from_im)
		# from_im_light = Image.fromarray(from_im_light)

		# from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
		from_im = from_im[:, :, [2,1,0]]
		from_im = Image.fromarray(from_im)


		to_path = self.target_paths[index]
		to_im = cv2.imread(to_path)
		# to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)


		to_im_gray = cv2.cvtColor(to_im, cv2.COLOR_BGR2GRAY)
		sketch = cv2.GaussianBlur(to_im_gray, (3, 3), 0)

		v = np.median(sketch)
		sigma = 0.33
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		sketch = cv2.Canny(sketch, lower, upper)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		sketch = cv2.dilate(sketch, kernel)

		sketch = np.expand_dims(sketch, axis=-1)
		sketch = np.concatenate([sketch, sketch, sketch], axis=-1)
		sketch[sketch < 0.5] = 0
		sketch[sketch >= 0.5] = 1
		# assert len(np.unique(sketch)) == 2

		to_im = to_im[:, :, [2,1,0]]
		to_im = Image.fromarray(to_im)

		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
			# from_im_light=self.source_transform(from_im_light)

		if self.train:
			if random.randint(0, 1):
				to_im = flip(to_im, 2)
				from_im = flip(from_im, 2)
				# from_im_light=flip(from_im_light,2)
				sketch = cv2.flip(sketch, 1)

		to_im=(to_im+1)*0.5
		from_im=(from_im+1)*0.5
    
		height = to_im.shape[1]
		width = to_im.shape[2]
		sketch[sketch == 255] = 1
		sketch = cv2.resize(sketch, (width, height))
		sketch = torch.from_numpy(sketch).permute(2, 0, 1)
		sketch = sketch[0:1, :, :]
		sketch = sketch.long()


		return from_im,to_im, sketch#sketch形状为(1,512,512)
