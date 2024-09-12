import svt

import math
import time
import numpy as np
import random
import h5py
import plot_utility as pu

import util
import gc
import load_data
import asyncio
import concurrent
from functools import partial

import load_data
import coil_est
import cupy as cp

import solvers
import grad

import prox

#base_path = '/media/buntess/OtherSwifty/Data/ITGADO/kallaren/'
base_path = '/media/buntess/OtherSwifty/Data/Test/COBRA_1/'
#base_path = '/media/buntess/OtherSwifty/Data/CO2/Pilot2/'
#base_path = '/media/buntess/OtherSwifty/Data/Garpen/Ena/'

#base_path = '/media/buntess/OtherSwifty/Data/ITGADO/gait9/'
#base_path = '/home/turbotage/Documents/4DRecon/'

async def get_smaps(lx=0.5, ls=0.0002, im_size=(320,320,320), load_from_zero = False, wexponent=0.75, translate = False, target_coils = 20, walsh = False):

	if load_from_zero:
		start = time.time()
		dataset = await load_data.load_flow_data(base_path + 'MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
		end = time.time()
		print(f"Load Time={end - start} s")
		
		start = time.time()
		dataset = await load_data.gate_ecg(dataset, 1)
		#dataset = await load_data.gate_CO2(dataset, 1)
		end = time.time()
		print(f"Gate Time={end - start} s")

		start = time.time()
		dataset['kdatas'], U = coil_est.coil_compress(dataset['kdatas'], 0, target_coils)
		end = time.time()
		print(f"Compress Time={end - start} s")

		start = time.time()
		dataset = await load_data.flatten(dataset)
		end = time.time()
		print(f"Flatten Time={end - start} s")

		start = time.time()
		dataset = await load_data.crop_kspace(dataset, im_size)
		end = time.time()
		print(f"Crop Time={end - start} s")

		if translate:
			start = time.time()
			dataset = await load_data.translate(dataset, (-2*im_size[0]/10, 0, 0))
			end = time.time()
			print(f"Translate Time={end - start} s")


		maxval = max(np.max(np.abs(kd)) for kd in dataset['kdatas'])
		for kd in dataset['kdatas']:
			kd[:] /= maxval

		maxval = max(np.max(np.abs(wd)) for wd in dataset['weights'])
		for wd in dataset['weights']:
			wd[:] /= maxval

		start = time.time()
		load_data.save_processed_dataset(dataset, base_path + 'dataset.h5')
		end = time.time()
		print(f"Save Dataset Time={end - start} s")
	else:
		start = time.time()
		dataset = load_data.load_processed_dataset(base_path + 'dataset.h5')
		end = time.time()
		print(f"Load Dataset Time={end - start} s")

	#smaps = await coil_est.walsh(list([dataset['coords'][0]]), list([dataset['kdatas'][0]]), list([dataset['weights'][0]]), imsize)


	
	#filename = f'/media/buntess/OtherSwifty/Data/COBRA191/reconed_lx{lx}_ls{ls}.h5'
	#print('Save')
	#with h5py.File(filename, 'w') as f:
	#	f['smaps'] = smaps
		

	#dataset['weights'] = [(w / w.max())**wexponent for w in dataset['weights']]



	dataset['weights'] = [(w)**wexponent for w in dataset['weights']]



	smaps, image = await coil_est.low_res_sensemap(dataset['coords'][0], dataset['kdatas'][0], dataset['weights'][0], im_size,
									  tukey_param=(0.95, 0.95, 0.95), exponent=3)
	
	if walsh:
		coil_images, ref_c = coil_est.create_coil_images(list([dataset['coords'][0]]), list([dataset['kdatas'][0]]), list([dataset['weights'][0]]), imsize)
		
		start = time.time()
		smaps = coil_est.walsh_cpu(coil_images, ref_c, np.array([9,9,9]))
		smaps = coil_est.sos_normalize(smaps)
		end = time.time()
		print(f"Walsh Took={end - start}")

	

	#pu.image_nd(smaps)

	


	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 1, "imsize": im_size, "typehint": "full"}

	do_isense = False
	if do_isense:
		smaps, image, alpha_i, resids = await coil_est.isense(image, smaps, 
			cp.array(dataset['coords'][0]), 
			cp.array(dataset['kdatas'][0]), 
			cp.array(dataset['weights'][0]),
			devicectxdict, 
			iter=[5,[4,7]],
			lamda=[lx, ls])
	else:
		async def inormal(imgnew):
			return await grad.gradient_step_x(smaps, imgnew, [dataset['coords'][0]], None, [dataset['weights'][0]], None, [devicectxdict])

		alpha_i = 1 / await solvers.max_eig(np, inormal, util.complex_rand(image[0,...][None,...].shape, xp=np), 8)


	image = np.repeat(image, 5, axis=0)


	filename = base_path + f'reconed_lowres_{im_size[0]}.h5'
	print('Save')
	with h5py.File(filename, 'w') as f:
		f.create_dataset('image', data=image)
		f.create_dataset('smaps', data=smaps)
		f.create_dataset('coil_comp', data=U)
	# del ....
	cp.get_default_memory_pool().free_all_blocks()



async def run_framed(niter, nframes, smapsPath, load_from_zero=True, imsize = (320,320,320), wexponent=0.75, lambda_n=1e-3, lambda_t=1e-5, 
					 translate = False, target_coils = 20, blk_sz = 4, pythonMaps = False, walsh = False):
	

	if load_from_zero:
		start = time.time()
		dataset = await load_data.load_flow_data(base_path + 'MRI_Raw.h5', gating_names=['TIME_E0', 'ECG_E0'])
		end = time.time()
		print(f"Load Time={end - start} s")

		start = time.time()
		U_cc = load_data.load_coil_mat(smapsPath)
		dataset['kdatas'] = coil_est.coil_compress_framed(dataset['kdatas'], U_cc, target_coils)
		end = time.time()
		print(f"Compress Time={end - start} s")
		concurrent
		start = time.time()
		dataset = await load_data.gate_ecg(dataset, nframes)
		#dataset = await load_data.gate_CO2(dataset, nframes)
		end = time.time()
		print(f"Gate Time={end - start} s")

		start = time.time()
		dataset = await load_data.flatten(dataset)
		end = time.time()
		print(f"Flatten Time={end - start} s")

		start = time.time()
		dataset = await load_data.crop_kspace(dataset, imsize)
		end = time.time()
		print(f"Crop Time={end - start} s")

		if translate:
			start = time.time()
			dataset = await load_data.translate(dataset, (-2*imsize[0]/10, 0, 0))
			end = time.time()
			print(f"Translate Time={end - start} s")


		maxval = max(np.max(np.abs(kd)) for kd in dataset['kdatas'])
		for kd in dataset['kdatas']:
			kd[:] /= maxval
		

		maxval = max(np.max(np.abs(wd)) for wd in dataset['weights'])
		for wd in dataset['weights']:
			wd[:] /= maxval
	

		start = time.time()
		load_data.save_processed_dataset(dataset, base_path + 'dataset_framed.h5')
		end = time.time()
		print(f"Save Dataset Time={end - start} s")
	else:
		start = time.time()
		dataset = load_data.load_processed_dataset(base_path + 'dataset_framed.h5')
		end = time.time()
		print(f"Load Dataset Time={end - start} s")
		U_cc = load_data.load_coil_mat(smapsPath)



	# Load smaps and full image
	smaps, image = load_data.load_smaps_image(smapsPath)
	if not(pythonMaps):
		smaps = load_data.load_orchestra_smaps(base_path + f'SenseMaps_{imsize[0]}.h5', 44, target_coils, imsize, U_cc)


	



	#dataset['weights'] = [(w / w.max())**wexponent for w in dataset['weights']]
	dataset['weights'] = [(w)**wexponent for w in dataset['weights']]
	

	#devicectx = grad.DeviceCtx(cp.cuda.Device(0), 2, imsize, "full")
	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 1, "imsize": imsize, "typehint": "framed"}
	
	#image = np.repeat(image, nframes, axis=0)
	image = np.tile(image, (nframes, 1, 1, 1))

	async def inormal(imgnew):
		return await grad.gradient_step_x(smaps, imgnew, [dataset['coords'][0]], None, [dataset['weights'][0]], None, [devicectxdict])

	alpha_i = 0.5 / await solvers.max_eig(np, inormal, util.complex_rand(image[0,...][None,...].shape, xp=np), 30)
	#dataset['weights'] = [(w)*alpha_i for w in dataset['weights']]
	#alpha_i = 

	print(alpha_i)
	async def gradx(ximg, a):
		normlist = await grad.gradient_step_x(smaps, ximg, dataset['coords'], dataset['kdatas'], dataset['weights'],
				a, [devicectxdict], calcnorm=True)
		
		print(f'Error = {np.array([a.item() for a in normlist[0]]).sum()}')
		

	#proxx = prox.svtprox(base_alpha=lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
	#proxx = prox.timefftprox(base_alpha=lambda_n)

	# if False:
	# 	proxx = prox.average_waveletLLR_prox(base_alpha_wave = lambda_t, base_alpha_LLR = lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
		
	# 	filename = base_path + f'long_run/reconed_wavelet_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n}_t{lambda_t:.2e}_'
	# elif False:
	# 	proxx = prox.average_DCT_LLR_prox(base_alpha_dct = lambda_t, base_alpha_LLR = lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
		
	# 	filename = base_path + f'long_run/reconed_dct_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2f}_llr{lambda_n:.8f}_t{lambda_t:.8f}_'
	if lambda_t > 5e-7:
		proxx = prox.composition_waveletLLR_prox(base_alpha_wave = lambda_t, base_alpha_LLR = lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
		
		filename = base_path + f'long_run/reconed_wavelet_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_t{lambda_t:.2e}_'
	else:
		proxx = prox.svtprox(base_alpha=lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
		if pythonMaps:
			if walsh:
				filename = base_path + f'long_run/{imsize[0]}/walsh/reconed_LLR_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_'
			else:
				filename = base_path + f'long_run/{imsize[0]}/LR/reconed_LLR_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_'
		else:
			filename = base_path + f'long_run/{imsize[0]}/reconed_LLR_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_'
	end


	await solvers.fista(np, image, alpha_i, gradx, proxx, niter, saveImage=True, fileName=filename)
	cp.get_default_memory_pool().free_all_blocks()

	

	del image, smaps, dataset
	gc.collect()




if __name__ == "__main__":
	
	do_trans = False
	target_coils = 20
	num_frames = 40

	lambda_x = 0.05 #round(10**(random.uniform(0, -4)), 5)
	lambda_s = round(10**(random.uniform(-2, -6)), 7)

	

	wexponent = [1]
	resolutions = [160]

	pythonMaps = False
	useWalsh = True
	# lambda_t = np.logspace(-2, -1, 2, 10)
	# lambda_llr = np.logspace(-6, -5, 2, 10)

	blocks = [4]
	lambda_llr = [1e-2, 1e-3, 1e-5, 1e-4, 1e-6]
	
	blocks, lambda_llr = np.meshgrid(blocks, lambda_llr)
	blocks = blocks.flatten()
	lambda_llr = lambda_llr.flatten()




	I = np.arange(lambda_llr.shape[0])
	#I = I[1:]
	#I = [4, 1]
	# 0, 2, 4, 1

	print(blocks)
	print(lambda_llr)
	print(I)

	for res in resolutions:

		imsize = (res,res,res)
		i = 0
		asyncio.run(get_smaps(lambda_x, lambda_s, imsize, True, wexponent=1, translate=do_trans, target_coils=target_coils, walsh=useWalsh))

		for wexp in wexponent:
			
			for idx in I:

				l = lambda_llr[idx]
				blk_sz = blocks[idx]
			
				print(f'Iteration number: {i}')
				
				cp.get_default_memory_pool().free_all_blocks()

				sPath = base_path + f'reconed_lowres_{res}.h5' 


				asyncio.run(run_framed(niter=200, nframes=num_frames, smapsPath=sPath, load_from_zero=False if i != 0 else True, imsize=imsize,  
							wexponent=wexp, lambda_n=l, lambda_t=0,  translate=do_trans, target_coils = target_coils, blk_sz=blk_sz, pythonMaps=pythonMaps, 
							walsh = useWalsh))
					
				# asyncio.run(run_framed_time(niter=100, nframes=num_frames, smapsPath=sPath, load_from_zero=False if i != 1 else True, imsize=imsize,  
				# 			  wexponent=wexp, lambda_n=l,  translate=do_trans, target_coils = target_coils, blk_sz=blk_sz))
				i += 1