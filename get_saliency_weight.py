import numpy as np
import os
from PIL import Image

# tile 960 x 672 -> 160 x 112, which is 6x6 tiles
# tile 1920 x 1080 -> 320 x 180, which is 6x6 tiles
img_width = 1920
img_height = 1080
patch_width = 320
patch_height = 180

height_patch_num = 6
width_patch_num = 6

videos = ['Haul','How-to','Product-review','Skit','Vlog','Unboxing','Game','Challenge']


for video in videos:

	src_path = 'output/all_video/'+video+'_LR/'
	des_path = 'saliency_weight/'

	delta = 1e-3

	def get_patch(img, idx, patch_height, patch_width):
		j = idx // height_patch_num
		i = idx % width_patch_num 
		return img[patch_height*j:patch_height*(j+1),patch_width*i:patch_width*(i+1)]


	m1_arr = []
	m2_arr = []
	m3_arr = []

	for img in sorted(os.listdir(src_path)):
		print(img)
		sal_map = Image.open(src_path+img)
		sal_map = sal_map.resize((img_width, img_height),resample=Image.BICUBIC)
		sal_map.load()
		sal_map = np.asarray(sal_map, dtype='float32')
		sal_map = sal_map/255.0
		m1_patches = []
		m2_patches = []
		m3_patches = []
		for j in range(height_patch_num):
			for i in range(width_patch_num):
				patch = sal_map[patch_height*j:patch_height*(j+1),patch_width*i:patch_width*(i+1)]
				patch_sum = patch.sum()
				m2_patches.append(patch_sum)
				if(patch_sum == 0):
					patch_sum += delta
				m1_patches.append(patch_sum)

		m1_patches = np.asarray(m1_patches)
		m2_patches = np.asarray(m2_patches)

		# method 1 (x+delta)/sum(x)
		m1_weight = m1_patches/np.sum(m1_patches)
		m1_arr.append(m1_weight)
		# print("------------M1---------------------\n")
		# print(m1_weight)
		# print(m1_weight.sum())

		# method 2 softmax, after min-max
		m2_patches = (m2_patches-np.min(m2_patches))/(np.max(m2_patches)-np.min(m2_patches))
		m2_weight = np.exp(m2_patches)/np.sum(np.exp(m2_patches))
		m2_arr.append(m2_weight)
		# print("------------M2---------------------\n")
		# print(m2_weight)
		# print(m2_weight.sum())

		# method 3, after softmax, use the (value+delta)/sum(value)
		for tt in m2_patches:
			if(tt == 0):
				tt += delta
				m3_patches.append(tt)
			else:
				m3_patches.append(tt)
		m3_patches = np.asarray(m3_patches)
		m3_weight = m3_patches/np.sum(m3_patches)
		m3_arr.append(m3_weight)
		# print("------------M3---------------------\n")
		# print(m3_weight)
		# print(m3_weight.sum())
		# break


	m1_arr = np.asarray(m1_arr)
	m2_arr = np.asarray(m2_arr)
	m3_arr = np.asarray(m3_arr)
	print(m1_arr.shape)
	# print(m2_arr.shape)
	# print(m3_arr.shape)
	np.save(des_path+str(delta)+'_'+video+'_m1_res.npy', m1_arr)
	np.save(des_path+str(delta)+'_'+video+'_m2_res.npy', m2_arr)
	np.save(des_path+str(delta)+'_'+video+'_m3_res.npy', m3_arr)

