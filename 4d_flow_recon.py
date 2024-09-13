import svt

import math
import time
import numpy as np
import random
import h5py
import plot_utility as pu
import post_processing_p
from post_processing_p import PostP_4DFlow

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

base_path = '/proj/nobackup/hpc2n2024-107/data/subject1/'




async def run_framed(niter, nframes, smapsPath, imsize = (320,320,320), wexponent=0.75, lambda_n=1e-3, lambda_t=1e-5, 
					target_coils = 20, blk_sz = 4, pythonMaps = False):
	


	start = time.time()
	dataset = load_data.load_processed_dataset(base_path + 'dataset_framed_{imsize[0]}.h5')
	end = time.time()
	print(f"Load Dataset Time={end - start} s")
	U_cc = load_data.load_coil_mat(smapsPath)



	# Load smaps and full image
	smaps, image = load_data.load_smaps_image(smapsPath)
	if not(pythonMaps):
		smaps = load_data.load_orchestra_smaps(base_path + f'SenseMaps_{imsize[0]}.h5', 44, target_coils, imsize, U_cc)


	dataset['weights'] = [(w)**wexponent for w in dataset['weights']]
	

	devicectxdict = {"dev": cp.cuda.Device(0), "ntransf": 1, "imsize": imsize, "typehint": "framed"}
	
	image = np.tile(image, (nframes, 1, 1, 1))

	async def inormal(imgnew):
		return await grad.gradient_step_x(smaps, imgnew, [dataset['coords'][0]], None, [dataset['weights'][0]], None, [devicectxdict])

	alpha_i = 0.5 / await solvers.max_eig(np, inormal, util.complex_rand(image[0,...][None,...].shape, xp=np), 30)


	print(alpha_i)
	async def gradx(ximg, a):
		normlist = await grad.gradient_step_x(smaps, ximg, dataset['coords'], dataset['kdatas'], dataset['weights'],
				a, [devicectxdict], calcnorm=True)
		
		print(f'Error = {np.array([a.item() for a in normlist[0]]).sum()}')
		

	if lambda_t > 5e-7:
		proxx = prox.composition_waveletLLR_prox(base_alpha_wave = lambda_t, base_alpha_LLR = lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
		
		filename = base_path + f'reconed_vels/reconed_wavelet_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_t{lambda_t:.2e}_'
	else:
		proxx = prox.svtprox(base_alpha=lambda_n, blk_shape=np.array([blk_sz, blk_sz, blk_sz]), blk_strides=np.array([blk_sz, blk_sz, blk_sz]), block_iter=4)
		if pythonMaps:
			filename = base_path + f'reconed_vels/{imsize[0]}/low_res_mps/reconed_LLR_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_'
		else:
			filename = base_path + f'reconed_vels/{imsize[0]}/reconed_LLR_framed{nframes}_blk{blk_sz}_wexp{wexponent:.2e}_llr{lambda_n:.2e}_'
	end


	await solvers.fista(np, image, alpha_i, gradx, proxx, niter, saveImage=False, fileName=filename)
	cp.get_default_memory_pool().free_all_blocks()

	# Calculate velocities
	image = image.reshape(nframes, 5, imsize[0], imsize[0], imsize[0])
	venc = 1100


		
	post4DFlow = PostP_4DFlow(venc, image)
	post4DFlow.solve_velocity()
	post4DFlow.update_cd()

	post4DFlow.correct_background_phase()
	post4DFlow.update_cd()

	filename = filename + 'vels_and_cd.h5'

	with h5py.File(filename, 'w') as f:
		f.create_dataset('vel', data=post4DFlow.vel)
		f.create_dataset('cd', data=post4DFlow.cd)
	

	del image, smaps, dataset
	gc.collect()




if __name__ == "__main__":
	
	target_coils = 20
	num_frames = 40
	

	wexponent = [1]
	resolutions = [160]

	pythonMaps = False


	blocks = [4]
	lambda_llr = [1e-2, 1e-3, 1e-5, 1e-4, 1e-6]
	
	blocks, lambda_llr = np.meshgrid(blocks, lambda_llr)
	blocks = blocks.flatten()
	lambda_llr = lambda_llr.flatten()




	I = np.arange(lambda_llr.shape[0])


	print(blocks)
	print(lambda_llr)
	print(I)

	for res in resolutions:

		imsize = (res,res,res)
		i = 0

		for wexp in wexponent:
			
			for idx in I:

				l = lambda_llr[idx]
				blk_sz = blocks[idx]
			
				print(f'Iteration number: {i}')
				
				cp.get_default_memory_pool().free_all_blocks()

				sPath = base_path + f'reconed_lowres_{res}.h5' 


				asyncio.run(run_framed(niter=200, nframes=num_frames, smapsPath=sPath, imsize=imsize,  wexponent=wexp, lambda_n=l, 
							lambda_t=0, target_coils = target_coils, blk_sz=blk_sz, pythonMaps=pythonMaps))
					
				i += 1