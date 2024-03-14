import math
import numba as nb
import cupy as cp
import numpy as np
import time
import gc

import asyncio
import concurrent

@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def block_fetcher_3d_numba(input, iter, shifts, br, Sr, bshape, bstrides, num_encodes, num_frames):

	bx = br[0]
	by = br[1]
	bz = br[2]

	Sx = Sr[0]
	Sy = Sr[1]
	Sz = Sr[2]

	shiftz = shifts[0, iter]
	shifty = shifts[1, iter]
	shiftx = shifts[2, iter]

	large_block = np.empty((bx * by * bz, bshape[0]*bshape[1]*bshape[2]*num_encodes, num_frames), dtype=np.complex64)

	
	for nz in nb.prange(bz):
		sz = nz * bstrides[0] + shiftz
		ez = sz + bshape[0]

		for ny in range(by):
			sy = ny * bstrides[1] + shifty
			ey = sy + bshape[1]

			for nx in range(bx):
				sx = nx * bstrides[2] + shiftx
				ex = sx + bshape[2]

				block_counter = nx + ny*bx + nz*by*bx
				for tframe in range(num_frames):
					count = 0
					for encode in range(num_encodes):
						store_pos = int(tframe * num_encodes + encode)
						for x in range(sx, ex):
							for y in range(sy, ey):
								for z in range(sz, ez):
									large_block[block_counter, count, tframe] = input[store_pos, x % Sx, y % Sy, z % Sz]
									count += 1

	return large_block

#def block_fetcher_3d(input, iter, shifts, br, Sr, bshape, bstrides, num_encodes, num_frames):
#    return block_fetcher_3d_numba(input, iter, shifts, br, Sr, bshape, bstrides, num_encodes, num_frames)

@nb.jit(nopython=True, cache=True, parallel=True, nogil=True)
def block_pusher_3d_numba(output, large_block, iter, shifts, br, Sr, bshape, bstrides, num_encodes, num_frames, scale):
	bx = br[0]
	by = br[1]
	bz = br[2]

	Sx = Sr[0]
	Sy = Sr[1]
	Sz = Sr[2]

	shiftz = shifts[0, iter]
	shifty = shifts[1, iter]
	shiftx = shifts[2, iter]

	for nz in nb.prange(bz):
		sz = nz * bstrides[0] + shiftz
		ez = sz + bshape[0]

		for ny in range(by):
			sy = ny * bstrides[1] + shifty
			ey = sy + bshape[1]

			for nx in range(bx):
				sx = nx * bstrides[2] + shiftx
				ex = sx + bshape[2]

				# Put block back
				block_counter = nx + ny*bx + nz*by*bx
				for tframe in range(num_frames):
					count = 0
					for encode in range(num_encodes):
						store_pos = int(tframe * num_encodes + encode)
						for x in range(sx, ex):
							for y in range(sy, ey):
								for z in range(sz, ez):
									output[store_pos, x % Sx, y % Sy, z % Sz] += scale*large_block[block_counter, count, tframe]
									count += 1



#def block_pusher_3d(output, large_block, iter, shifts, br, Sr, bshape, bstrides, num_encodes, num_frames, scale):
#    return block_pusher_3d_numba(output, large_block, iter, shifts, br, Sr, bshape, bstrides, num_encodes, num_frames, scale)

#@nb.jit(nopython=True, cache=True, parallel=True)'
@nb.jit(nopython=True, cache=True, nogil=True)
def spatial_block_fetcher_3d_numba(input, shifts, br, Sr, bshape, bstrides, dir):
	bx = br[0]
	by = br[1]
	bz = br[2]

	Sx = Sr[0]
	Sy = Sr[1]
	Sz = Sr[2]


	if dir == 'x':
		large_block = np.empty((bx * by * bz, bshape[1]*bshape[2], bshape[0]), dtype=np.complex64)
	elif dir == 'y':
		large_block = np.empty((bx * by * bz, bshape[0]*bshape[2], bshape[1]), dtype=np.complex64)
	elif dir == 'z':
		large_block = np.empty((bx * by * bz, bshape[0]*bshape[1], bshape[2]), dtype=np.complex64)

	for nx in range(bx):
		sx = nx * bstrides[0] + shifts[0]
		ex = sx + bshape[0]

		for ny in range(by):
			sy = ny * bstrides[1] + shifts[1]
			ey = sy + bshape[1]

			for nz in range(bz): #nb.prange(bz):
				sz = nz * bstrides[2] + shifts[2]
				ez = sz + bshape[2]

				dircount = 0
				block_counter = nx*by*bz + ny*bz + nz
				if dir == 'x':
					for x in range(sx, ex):
						count = 0
						for y in range(sy, ey):
							for z in range(sz, ez):
								large_block[block_counter, count, dircount] = input[x % Sx, y % Sy, z % Sz]
								count += 1
						dircount += 1
				elif dir == 'y':
					for y in range(sy, ey):
						count = 0
						for x in range(sx, ex):
							for z in range(sz, ez):
								large_block[block_counter, count, dircount] = input[x % Sx, y % Sy, z % Sz]
								count += 1
						dircount += 1
				elif dir == 'z':
					for z in range(sz, ez):
						count = 0
						for x in range(sx, ex):
							for y in range(sy, ey):
								large_block[block_counter, count, dircount] = input[x % Sx, y % Sy, z % Sz]
								count += 1
						dircount += 1


	return large_block

@nb.jit(nopython=True, cache=True, nogil=True)
def spatial_block_pusher_3d_numba(output, large_block, shifts, br, Sr, bshape, bstrides, scale, dir):
	bx = br[0]
	by = br[1]
	bz = br[2]

	Sx = Sr[0]
	Sy = Sr[1]
	Sz = Sr[2]

	for nx in range(bx):
		sx = nx * bstrides[0] + shifts[0]
		ex = sx + bshape[0]

		for ny in range(by):
			sy = ny * bstrides[1] + shifts[1]
			ey = sy + bshape[1]

			for nz in range(bz):
				sz = nz * bstrides[2] + shifts[2]
				ez = sz + bshape[2]

				dircount = 0
				block_counter = nx*by*bz + ny*bz + nz
				# Pushes x block
				if dir == 'x':
					for x in range(sx, ex):
						count = 0
						for y in range(sy, ey):
							for z in range(sz, ez):
								output[x % Sx, y % Sy, z % Sz] += scale*large_block[block_counter, count, dircount]
								count += 1
						dircount += 1
				elif dir == 'y':
					for y in range(sy, ey):
						count = 0
						for x in range(sx, ex):
							for z in range(sz, ez):
								output[x % Sx, y % Sy, z % Sz] += scale*large_block[block_counter, count, dircount]
								count += 1
						dircount += 1
				elif dir == 'z':
					for z in range(sz, ez):
						count = 0
						for x in range(sx, ex):
							for y in range(sy, ey):
								output[x % Sx, y % Sy, z % Sz] += scale*large_block[block_counter, count, dircount]
								count += 1
						dircount += 1



@cp.fuse(kernel_name='thresh_blocks_cupy')
def softthresh_func(s, lamda):
	return cp.maximum(0, (cp.abs(s) - lamda))

def thresh_blocks(lblock, lamda, max_run_blocks):
	stream = cp.cuda.Stream(non_blocking=True)

	runs = math.ceil(lblock.shape[0] / max_run_blocks)
	with stream:
		for run in range(runs):
			start = run*max_run_blocks
			end = min(start + max_run_blocks, lblock.shape[0])

			cu_lblock = cp.array(lblock[start:end,...])
			u, s, v = cp.linalg.svd(cu_lblock, full_matrices=False)
			s = softthresh_func(s, lamda)
			cu_lblock = u @ (s[...,None] * v)
			lblock[start:end,...] = cu_lblock.get()

	del cu_lblock, u, s, v
	cp.get_default_memory_pool().free_all_blocks()

	return lblock
			






async def my_svt3(output, input, lamda, blk_shape, blk_strides, block_iter, num_encodes):
	imsize = input.shape[1:]
	br = np.array([imsize[0] // blk_strides[0], imsize[1] // blk_strides[1], imsize[2] // blk_strides[2]])
	Sr = np.array([int(imsize[0]), int(imsize[1]), int(imsize[2])])

	scale = float(1.0 / block_iter)

	num_frames = int(input.shape[0]/num_encodes)

	shifts = np.zeros((3, block_iter), np.int32)
	for d in range(3):
		for biter in range(block_iter):
			shifts[d,biter] = np.random.randint(blk_shape[d])

	#lock = asyncio.Lock()
	#loop = asyncio.get_event_loop()
	#executor = concurrent.futures.ThreadPoolExecutor(max_workers=(block_iter))

	def fetch_and_thresh(iter):
		start = time.time()
		large_block =  block_fetcher_3d_numba(input, iter, shifts, br, Sr, blk_shape, blk_strides, num_encodes, num_frames)
		end = time.time()
		print(f"Fetcher Time = {end - start}")

		start = time.time()
		large_block = thresh_blocks(large_block, lamda, 50)
		end = time.time()
		print(f"SoftThresh Time = {end - start}")

		return large_block

	#futures = []
	#for iter in range(block_iter):
	#	futures.append(loop.run_in_executor(executor, fetch_and_thresh, iter))

	for iter in range(block_iter):
		
		#large_block = await futures[iter]

		
		large_block = fetch_and_thresh(iter)
		start = time.time()
		block_pusher_3d_numba(output, large_block, iter, shifts, br, Sr, blk_shape, blk_strides, num_encodes, num_frames, scale)
		end = time.time()
		print(f"Pusher Time = {end - start}")


	return output




async def my_spatial_svt3(output, input, lamda, blk_shape, blk_strides, block_iter):
	imsize = input.shape[1:]
	br = np.array([imsize[0] // blk_strides[0], imsize[1] // blk_strides[1], imsize[2] // blk_strides[2]])
	Sr = np.array([int(imsize[0]), int(imsize[1]), int(imsize[2])])

	scale = float(1.0 / (3 * block_iter))

	shifts = np.zeros((3, 3, block_iter), np.int32)
	for d in range(3):
		for axis in range(3):
			for biter in range(block_iter):
				shifts[d,axis,biter] = np.random.randint(blk_shape[d])

	loop = asyncio.get_event_loop()
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

	def fetch_and_thresh(frame, local_shifts, dir):

		start = time.time()
		large_block =  spatial_block_fetcher_3d_numba(np.ascontiguousarray(input[frame,...]), local_shifts, br, Sr, blk_shape, blk_strides, dir)
		end = time.time()
		#print(f"Fetcher Time = {end - start}")

		start = time.time()
		large_block = thresh_blocks(large_block, lamda, 200)
		end = time.time()
		#print(f"SoftThresh Time = {end - start}")

		start = time.time()
		spatial_block_pusher_3d_numba(output[frame,...], large_block, local_shifts, br, Sr, blk_shape, blk_strides, scale, dir)
		end = time.time()
		#print(f"Pusher Time = {end - start}")
		#print(f"Frame = {frame}, iter = {iter}, dir = {dir}")

	async def run_frames(dir, local_shifts):
		futures = []
		for frame in range(input.shape[0]):
			futures.append(loop.run_in_executor(executor, fetch_and_thresh, frame, local_shifts, dir))
	
			if len(futures) > 4:
				await futures[0]
				del futures[0]
	
		for fut in futures:
			await fut

	#def run_frames(dir, local_shifts):
	#	for frame in range(input.shape[0]):
	#		fetch_and_thresh(frame, local_shifts, dir)

	for iter in range(block_iter):
		# X-dir
		local_shifts = np.array([shifts[0, 0, iter], shifts[1, 0, iter], shifts[2, 0, iter]])
		await run_frames('x', local_shifts)
		# Y-dir
		local_shifts = np.array([shifts[0, 1, iter], shifts[1, 1, iter], shifts[2, 1, iter]])
		await run_frames('y', local_shifts)
        # Z-dir
		local_shifts = np.array([shifts[0, 2, iter], shifts[1, 2, iter], shifts[2, 2, iter]])
		await run_frames('z', local_shifts)

	cp.get_default_memory_pool().free_all_blocks()
	return output



	by = input.shape[1] // blk_strides[0]
	bx = input.shape[2] // blk_strides[1]

	Sy = int(input.shape[1])
	Sx = int(input.shape[2])

	scale = float(1.0 / block_iter)

	num_frames = int(input.shape[0]/num_encodes)
	bmat_shape = (num_encodes*blk_shape[0] * blk_shape[1], num_frames)

	shifts = np.zeros((2, block_iter), np.int32)
	for d in range(2):
		for biter in range(block_iter):
			shifts[d,biter] = np.random.randint(blk_shape[d])

	for iter in range(block_iter):
		#print('block iter = ',iter)
		shiftx = shifts[0, iter]
		shifty = shifts[1, iter]

		for ny in nb.prange(by):
			sy = ny * blk_strides[0] + shifty
			ey = sy + blk_shape[0]

			for nx in range(bx):

				sx = nx * blk_strides[1] + shiftx
				ex = sx + blk_shape[1]

				block = np.zeros(bmat_shape, input.dtype)

				# Grab a block
				for tframe in range(num_frames):
					count = 0
					for encode in range(num_encodes):
						store_pos = int(tframe * num_encodes + encode)
						for j in range(sy, ey):
							for i in range(sx, ex):
								block[count, tframe] = input[store_pos, j % Sy, i % Sx]
								count += 1

				# Svd
				u, s, vh = np.linalg.svd(block, full_matrices=False)

				for k in range(u.shape[1]):

					# s[k] = max(s[k] - lamda, 0)
					abs_input = abs(s[k])
					if abs_input == 0:
						sign = 0
					else:
						sign = s[k] / abs_input

					s[k] = abs_input - lamda
					s[k] = (abs(s[k]) + s[k]) / 2
					s[k] = s[k] * sign

					for i in range(u.shape[0]):
						u[i, k] *= s[k]

				block = np.dot(u, vh)

				# Put block back
				for tframe in range(num_frames):
					count = 0
					for encode in range(num_encodes):
						store_pos = int(tframe * num_encodes + encode)
						for j in range(sy, ey):
							for i in range(sx, ex):
								output[store_pos, j % Sy, i % Sx] += scale*block[count, tframe]
								count += 1

	return output