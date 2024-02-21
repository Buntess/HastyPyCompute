import numpy as np
import cupy as cp
import cupyx as cpx
import cupyx.scipy as cpxsp
import cufinufft
import math
import util

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
				self.forward_plan = cufinufft.Plan(nufft_type=2, n_modes=imshape, 
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
		if self.type == "none":
			self.coord = coord

		if coord.shape[0] == 1:
			self.forward_plan.setpts(x=coord[0,:])
		elif coord.shape[0] == 2:
			self.forward_plan.setpts(x=coord[0,:], y=coord[1,:])
		elif coord.shape[0] == 3:
			self.forward_plan.setpts(x=coord[0,:], y=coord[1,:], z=coord[2,:])
		else:
			raise ValueError(f"Invalid number of coordinates ({coord.shape[0]})")

	def setpts_backward(self, coord):
		if self.type == "none":
			self.coord = coord

		if coord.shape[0] == 1:
			self.backward_plan.setpts(x=coord[0,:])
		elif coord.shape[0] == 2:
			self.backward_plan.setpts(x=coord[0,:], y=coord[1,:])
		elif coord.shape[0] == 3:
			self.backward_plan.setpts(x=coord[0,:], y=coord[1,:], z=coord[2,:])
		else:
			raise ValueError(f"Invalid number of coordinates ({coord.shape[0]})")

	def setpts(self, coord):
		if self.type == "none":
			self.coord = coord
		else:
			self.setpts_forward(coord)
			self.setpts_backward(coord)

	def forward_execute(self, input, out):
		if self.forward_plan is not None:
			self.forward_plan.execute(input, out)
			out *= self.normfactor
		elif self.type == "none":
			if self.coord.shape[0] == 1:
				cufinufft.nufft1d2(x=self.coord[0], data=input, out=out, eps=1e-4)
			elif self.coord.shape[0] == 2:
				cufinufft.nufft2d2(x=self.coord[0], y=self.coord[1], data=input, out=out, eps=1e-4)
			elif self.coord.shape[0] == 3:
				cufinufft.nufft3d2(x=self.coord[0], y=self.coord[1], z=self.coord[2], data=input, out=out, eps=1e-4)


	def backward_execute(self, input, out):
		if self.backward_plan is not None:
			input *= self.normfactor
			self.backward_plan.execute(input, out)
			#out *= self.normfactor
		elif self.type == "none":
			if self.coord.shape[0] == 1:
				cufinufft.nufft1d1(x=self.coord[0], data=input, n_modes=out.shape[1:], out=out, eps=1e-4)
			elif self.coord.shape[0] == 2:
				cufinufft.nufft2d1(x=self.coord[0], y=self.coord[1], data=input, n_modes=out.shape[1:], out=out, eps=1e-4)
			elif self.coord.shape[0] == 3:
				cufinufft.nufft3d1(x=self.coord[0], y=self.coord[1], z=self.coord[2], data=input, n_modes=out.shape[1:], out=out, eps=1e-4)


# Inputs shall be on CPU or GPU, computes x = x - alpha * S^HN^H(W(NSx-b)) or returns S^HN^HWN^HSx
async def device_gradient_step_x(smaps, images, coords, kdatas, weights, alpha, devicectx: DeviceCtx, calcnorm = False):
	
	if isinstance(devicectx, dict):
		devicectx = DeviceCtx(devicectx["dev"], devicectx["ntransf"], devicectx["imsize"], devicectx["typehint"])

	device = devicectx.device
	
	with device:
		ncoils = smaps.shape[0]
		numframes = images.shape[0]
		ntransf = devicectx.ntransf

		if ntransf is None:
			ntransf = ncoils
		if ncoils % ntransf != 0:
			raise ValueError(f"Number of smaps ({ncoils}) must be divisible by ntransf ({ntransf})")

		smaps_gpu = cp.array(smaps)

		if alpha is None:
			images_out = util.get_array_backend(images).zeros_like(images)

		cp.fuse(kernel_name='weights_and_kdata_func')
		def weights_and_kdata_func(kdmem, kd, w):
			return w*(kdmem - kd)

		cp.fuse(kernel_name='sum_smaps_func')
		def sum_smaps_func(imgmem, sin, ain):
			return ain * cp.sum(imgmem * cp.conj(sin), axis=0)

		normlist = []

		runs = int(ncoils / ntransf)
		for frame in range(numframes):
			image_frame = cp.array(images[frame,...], copy=False)
			weights_frame = cp.array(weights[frame], copy=False)
			coord_frame = cp.array(coords[frame], copy=False)

			if normlist is not None:
				normlist.append(0.0)

			devicectx.setpts(coord_frame)
			for run in range(runs):
				start = run * ntransf

				if kdatas is not None:
					kdata_frame = cp.array(kdatas[frame][start:start+ntransf,...], copy=False)

				kdatamem = cp.empty((ntransf,coord_frame.shape[1]), dtype=image_frame.dtype)

				locals = smaps_gpu[start:start+ntransf,...]

				imagemem = locals * image_frame

				cp.cuda.get_current_stream().synchronize()

				devicectx.forward_execute(imagemem, out=kdatamem)

				cp.cuda.get_current_stream().synchronize()

				if kdatas is not None:
					kdatamem = weights_and_kdata_func(kdatamem, kdata_frame, weights_frame)
					if normlist is not None:
						normlist[frame] += cp.linalg.norm(kdatamem)
				else:
					kdatamem *= weights_frame


				cp.cuda.get_current_stream().synchronize()

				devicectx.backward_execute(kdatamem, out=imagemem)

				cp.cuda.get_current_stream().synchronize()

				if alpha is None:
					if hasattr(images, 'device'):
						if images.device == devicectx.device:
							images_out[frame,...] += sum_smaps_func(imagemem, locals, 1.0)
						else:
							raise RuntimeError('images must reside on same device as devicectx or in cpu')
					else:
						images_out[frame,...] += sum_smaps_func(imagemem, locals, 1.0).get()
				else:
					if hasattr(images, 'device'):
						if images.device == devicectx.device:
							images[frame,...] -= sum_smaps_func(imagemem, locals, alpha)
						else:
							raise RuntimeError('images must reside on same device as devicectx or in cpu')
					else:
						images[frame,...] -= sum_smaps_func(imagemem, locals, alpha).get()

		if alpha is None:
			return images_out
		elif calcnorm:
			return normlist

async def gradient_step_x(smaps, images, coords, kdatas, weights, alpha, devicectxs: list, calcnorm = False):
	numframes = images.shape[0]

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
	for devindex, fpd in enumerate(frames_per_device):
		end = start + fpd
		futures.append(loop.run_in_executor(executor, device_gradient_step_x,
			smaps, 
			images[start:end,...], 
			coords[start:end], 
			None if kdatas is None else kdatas[start:end], 
			weights[start:end], 
			alpha, devicectxs[devindex], calcnorm))
		start = end


	normlist = []

	start = 0
	for i, fut in enumerate(futures):
		if alpha is None:
			end = start + frames_per_device[i]
			images_out[start:end] = await (await fut)
			start = end
		elif calcnorm:
			normlist.append(await (await fut))
		else:
			await (await fut)

	if alpha is None:
		return images_out
	elif calcnorm:
		return normlist





# Inputs shall be on GPU
async def device_gradient_step_s(smaps, images, coords, kdatas, weights, alpha, devicectx: DeviceCtx, calcnorm=False):
	with devicectx.device:
		ncoils = smaps.shape[0]
		ntransf = devicectx.ntransf


		cp.fuse(kernel_name='weights_and_kdata_func')
		def weights_and_kdata_func(kdmem, kd, w):
			return w*(kdmem - kd)

		if alpha is None:
			smaps_out = util.get_array_backend(smaps).empty_like(smaps)

		img = cp.array(images, copy=False)
		smp = cp.array(smaps, copy=False)

		norm = 0

		devicectx.setpts(cp.array(coords, copy=False))
		runs = int(ncoils / ntransf)
		for run in range(runs):
			start = run * ntransf

			if kdatas is not None:
				kd = cp.array(kdatas[start:start+ntransf,...], copy=False)

			kdmem = cp.empty((ntransf,coords.shape[1]), dtype=images.dtype)

			smem = img * smp[start:start+ntransf,...]

			devicectx.forward_execute(smem, kdmem)

			if kdatas is not None:
				kdmem = weights_and_kdata_func(kdmem, kd, weights)
				if calcnorm:
					norm += cp.linalg.norm(kdmem)
			else:
				kdmem *= weights

			devicectx.backward_execute(kdmem, out=smem)

			if alpha is None:
				if hasattr(smaps, 'device'):
					if smaps.device == devicectx.device:
						smaps_out[start:start+ntransf,...] = cp.conj(img) * smem
					else:
						raise RuntimeError('smaps must reside on same device as devicectx or in cpu')
				else:
					smaps_out[start:start+ntransf,...] = (cp.conj(img) * smem).get()
			else:
				if hasattr(smaps, 'device'):
					if smaps.device == devicectx.device:
						smaps[start:start+ntransf,...] -= alpha * cp.conj(img) * smem
					else:
						raise RuntimeError('smaps must reside on same device as devicectx or in cpu')
				else:
					smaps[start:start+ntransf,...] -= (alpha * cp.conj(img) * smem).get()


		if alpha is None:
			return smaps_out
		elif calcnorm:
			return norm

async def gradient_step_s(smaps, images, coords, kdatas, weights, alpha, devicectx: DeviceCtx, calcnorm=False):
	return await device_gradient_step_s(smaps, images, coords, kdatas, weights, alpha, devicectx, calcnorm)