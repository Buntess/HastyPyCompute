import numpy as np
import cupy as cp
import cupyx as cpx
import cupyx.scipy as cpxsp
import cufinufft
import math

import asyncio
import concurrent

class DeviceCtx:
	def __init__(self, 
				device: cp.cuda.Device | None, 
				ntransf: int,
				imshape: tuple[int],
				type = "",
				forward_plan: cufinufft.Plan | None = None, 
				backward_plan: cufinufft.Plan | None = None, 
			  ):
		
		self.device = device
		self.ntransf = ntransf
		self.imshape = imshape
		self.type = type

		self.normfactor = 1.0 / math.sqrt(math.prod(imshape))


		if forward_plan is None:
			if type == "full":
				self.forward_plan = cufinufft.Plan(nufft_type=2, n_modes=imshape, 
					n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
					gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
					gpu_device_id=device.id)
			elif type == "framed":
				self.forward_plan = cufinufft.Plan(nufft_type=2, n_modes=imshape, 
					n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=1.25,
					gpu_method=1, gpu_sort=1, gpu_kerevalmeth=0,
					gpu_device_id=device.id)
			elif type == "none":
				self.forward_plan = None
			else:
				self.forward_plan = cufinufft.Plan(ufft_type=2, n_modes=imshape, 
					n_trans=ntransf, eps=1e-4, dtype="complex64",
					gpu_device_id=device.id)
		else:
			self.forward_plan = forward_plan

		if backward_plan is None:
			if type == "full":
				self.backward_plan = cufinufft.Plan(nufft_type=1, n_modes=imshape,
					n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
					gpu_method=2,
					gpu_device_id=device.id)
			elif type == "framed":
				self.backward_plan = cufinufft.Plan(nufft_type=1, n_modes=imshape,
					n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
					gpu_method=2,
					gpu_device_id=device.id)
			elif type == "none":
				self.backward_plan = None
			else:
				self.backward_plan = cufinufft.Plan(nufft_type=1, n_modes=imshape,
					n_trans=ntransf, eps=1e-4, dtype="complex64", upsampfac=2.0,
					gpu_method=2,
					gpu_device_id=device.id)
		else:
			self.backward_plan = backward_plan

	def setpts_forward(self, coord):
		if coord.shape[0] == 1:
			self.forward_plan.setpts(x=coord[0,:])
		elif coord.shape[0] == 2:
			self.forward_plan.setpts(x=coord[0,:], y=coord[1,:])
		elif coord.shape[0] == 3:
			self.forward_plan.setpts(x=coord[0,:], y=coord[1,:], z=coord[2,:])
		else:
			raise ValueError(f"Invalid number of coordinates ({coord.shape[0]})")

	def setpts_backward(self, coord):
		if coord.shape[0] == 1:
			self.backward_plan.setpts(x=coord[0,:])
		elif coord.shape[0] == 2:
			self.backward_plan.setpts(x=coord[0,:], y=coord[1,:])
		elif coord.shape[0] == 3:
			self.backward_plan.setpts(x=coord[0,:], y=coord[1,:], z=coord[2,:])
		else:
			raise ValueError(f"Invalid number of coordinates ({coord.shape[0]})")
		
	def setpts(self, coord):
		self.setpts_forward(coord)
		self.setpts_backward(coord)

	def forward_execute(self, input, out):
		if self.forward_plan is not None:
			self.forward_plan.execute(input, out) * self.normfactor

	def backward_execute(self, input, out):
		if self.backward_plan is not None:
			self.backward_plan.execute(input, out) * self.normfactor


# Inputs shall be on CPU
async def device_gradient_step_x(smaps, images, coords, kdatas, weights, alpha, devicectx: DeviceCtx):
	with devicectx.device:
		ncoils = smaps.shape[0]
		numframes = images.shape[0]

		if ntransf is None:
			ntransf = ncoils
		if ncoils % ntransf != 0:
			raise ValueError(f"Number of smaps ({ncoils}) must be divisible by ntransf ({ntransf})")

		smaps_gpu = cp.array(smaps)

		if alpha is None:
			images_out = images.copy()

		cp.fuse(kernel_name='weights_and_kdata_func')
		def weights_and_kdata_func(kdmem, kd, w):
			return w*(kdmem - kd)
		
		cp.fuse(kernel_name='sum_smaps_func')
		def sum_smaps_func(imgmem, s, alpha):
			return alpha * cp.sum(imgmem * cp.conj(s))

		runs = int(ncoils / ntransf)
		for frame in range(numframes):
			image_frame = cp.array(images[i,...], copy=False)
			weights_frame = cp.array(weights[i], copy=False)
			coord_frame = cp.array(coords[i], copy=False)

			for run in range(runs):
				start = run * ntransf

				if kdatas is not None:
					kdata_frame = cp.array(kdatas[frame][start:start+ntransf,...], copy=False)
					kdatamem = cp.empty_like(kdata_frame)
				else:
					kdatamem = cp.empty((ntransf,coord_frame[1]), dtype=image_frame.dtype)

				locals = smaps_gpu[start:start+ntransf,...]

				imagemem = locals * image_frame

				devicectx.setpts(coord_frame)

				devicectx.forward_execute(imagemem, kdatamem)

				if kdatas is not None:
					kdatamem = weights_and_kdata_func(kdatamem, kdata_frame, weights_frame)
				else:
					kdatamem *= weights_frame

				devicectx.backward_execute(kdatamem, out=imagemem)
				
				if alpha is None:
					if images.device == devicectx.device:
						images[frame,...] = sum_smaps_func(imagemem, locals, 1.0)
					else:
						images[frame,...] = sum_smaps_func(imagemem, locals, 1.0).get()
				else:
					if images.device == devicectx.device:
						images[frame,...] -= sum_smaps_func(imagemem, locals, alpha)
					else:
						images[frame,...] -= sum_smaps_func(imagemem, locals, alpha).get()

async def gradient_step_x(smaps, images, coords, kdatas, weights, alpha, devicectxs: list[DeviceCtx]):
	numframes = images.shape[0]
	runners = numframes 

	# This is how many equally distributed frames per device we have
	frames_per_device = [numframes // len(devicectxs) for i in range(len(devicectxs))]
	leftover = numframes % len(devicectxs)

	# Distribute leftover frames as evenly as possible
	for i in range(leftover):
		devindex = i % len(devicectxs)
		frames_per_device[devindex] += 1

	loop = asyncio.get_event_loop()
	executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(devicectxs))

	if alpha is None:
		images_out = images.copy()

	futures = []
	start = 0
	for fpd in frames_per_device:
		end = start + fpd
		futures.append(loop.run_in_executor(executor, device_gradient_step_x,
			smaps[start:end,...], 
			images[start:end,...], 
			coords[start:end,...], 
			kdatas[start:end,...], 
			weights[start:end,...], 
			alpha, devicectxs[devindex]))
		start = end
	
	
	for fut in futures:
		await futures



# Inputs shall be on GPU
async def device_gradient_step_s(smaps, images, coords, kdatas, weights, alpha, devicectx: DeviceCtx):
	with devicectx.device:
		ncoils = smaps.shape[0]
		ntransf = devicectx.ntransf

		devicectx.setpts(coords)

		cp.fuse(kernel_name='weights_and_kdata_func')
		def weights_and_kdata_func(kdmem, kd, w):
			return w*(kdmem - kd)

		if alpha is None:
			smaps_out = cp.empty_like(smaps)

		runs = int(ncoils / ntransf)
		for run in range(runs):
			start = run * ntransf

			if kdatas is not None:
				kd = cp.array(kdatas[start:start+ntransf,...], copy=False)
				kdmem = cp.empty_like(kd)
			else:
				kdmem = cp.empty((ntransf,coords[1]), dtype=images.dtype)


			smem = images * smaps[start:start+ntransf,...]


			devicectx.forward_execute(smem, kdmem)

			if kdatas is not None:
				kdmem = weights_and_kdata_func(kdmem, kd, weights)
			else:
				kdmem *= weights

			devicectx.backward_execute(kdmem, out=smem)
			
			if alpha is None:
				smaps_out[start:start+ntransf,...] = cp.conj(images) * smem
			else:
				smaps[start:start+ntransf,...] -= alpha * cp.conj(images) * smem

		if alpha is None:
			return smaps_out