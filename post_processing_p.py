import cupy as cp
import h5py
import numpy as np
import math
import orthoslicer as ort
import plot_utility as pu

import gc


def constant_from_venc(venc):
	return (cp.pi / (venc * math.sqrt(3)))

def get_encode_matrix(k):
	Emat = k * cp.array(
		[
			#[ 0,  0,  0],
			[-1, -1, -1],
			[ 1,  1, -1],
			[ 1, -1,  1],
			[-1,  1,  1]
		], dtype=cp.float32)
	return Emat

def _enc_to_vel_linear(image, venc, nwraps):
	Emat = get_encode_matrix(constant_from_venc(venc))
	pEmat = cp.linalg.pinv(Emat)

	im_size = (image.shape[1], image.shape[2], image.shape[3])
	print("imsize: ", im_size)
	nvoxel = math.prod(list(im_size))

	base_phase = cp.angle(image[0,...])
	base_corrected = (image * cp.exp(-1j*base_phase)[None, ...]).reshape((5,nvoxel))


	phases = cp.angle(base_corrected) + 2 * cp.pi * nwraps.reshape((5,nvoxel))

	imageout = pEmat @ phases[1:,...]
	print(imageout.min())
	print(imageout.max())
	#print(venc * np.sqrt(3))

	mag = cp.mean(cp.abs(base_corrected), axis=0)[None,...]

	imageout = cp.concatenate([mag, imageout], axis=0)

	return (imageout, base_corrected)

def enc_to_vel_linear(images, venc, nwraps, plotAngles = False):
	if plotAngles:
		ort.image_nd(np.angle(images[:,0,...]).get())
		ort.image_nd(np.angle(images[:,1,...]).get())
		ort.image_nd(np.angle(images[:,2,...]).get())
		ort.image_nd(np.angle(images[:,3,...]).get())
		ort.image_nd(np.angle(images[:,4,...]).get())
		print('Is Contigous: ', np.is_conti)

	images_out = cp.empty((images.shape[0], 4, images.shape[2], images.shape[3], images.shape[4]))
	for i in range(images.shape[0]):
		print('Frame: ', i)
		image = images[i,...]
		images_out[i,...] = _enc_to_vel_linear(cp.array(image), venc, cp.array(nwraps[i, ...]))[0].reshape((4,images.shape[2], 
								  images.shape[3], images.shape[4]))
		
	if plotAngles:
		ort.image_nd(images_out[:,0,...].get())
		ort.image_nd(images_out[:,1,...].get())
		ort.image_nd(images_out[:,2,...].get())
		ort.image_nd(images_out[:,3,...].get())
	return images_out




class PostP_4DFlow:

	def __init__(self, venc, image):

		'Initialization'
		self.venc = venc  #m/s
		self.nwraps = np.zeros_like(np.real(image))
		
		# Matrices
		self.vel = None
		self.vel_mag = None
		self.cd = None
		self.mag = np.mean(np.abs(image), axis=1)
		self.image = image


	def solve_velocity(self):
		temp = enc_to_vel_linear(cp.array(self.image), self.venc, cp.array(self.nwraps))
		self.vel = temp[:, 1:, ...]
		
	
	def update_cd(self):

		vmag = cp.sqrt(cp.mean(cp.abs(cp.array(self.vel))**2 , 1))
		self.cd = (cp.array(self.mag)*cp.sin(math.pi/2.0*vmag/self.venc)).get()

		idx = cp.where(vmag > (self.venc/2))
		mag_temp = cp.array(self.mag)
		cd_temp = cp.array(self.cd)
		cd_temp[idx] = mag_temp[idx]
		self.cd = cd_temp.get()

	def update_mag(self, image, axis):

		self.mag = np.mean(np.abs(image), axis=axis)


	def correct_background_phase(self, mag_thresh=0.08, cd_thresh=0.3, fit_order=2):
		
		# Average and max
		mag_avg = cp.mean(cp.array(self.mag), axis=0)
		cd_avg = cp.mean(cp.array(self.cd), axis=0)

		max_mag = cp.max(mag_avg)
		max_cd = cp.max(cd_avg)

		# Get the number of coeficients
		range_temp = cp.arange(0, fit_order+1)
		pz,py,px = cp.meshgrid(range_temp, range_temp, range_temp)
		idx = cp.where( (px+py+pz) <= fit_order )
		px = px[idx]
		py = py[idx]
		pz = pz[idx]
		N = len(px)

		AhA = cp.zeros([N, N], dtype=cp.float32)
		AhBx = cp.zeros([N, 1], dtype=cp.float32)
		AhBy = cp.zeros([N, 1], dtype=cp.float32)
		AhBz = cp.zeros([N, 1], dtype=cp.float32)


		# Now gather terms (Nt x Nz x Ny x Nx x 3 )
		z, y, x = cp.meshgrid(cp.linspace(-1, 1, self.vel.shape[2]),
								cp.linspace(-1, 1, self.vel.shape[3]),
								cp.linspace(-1, 1, self.vel.shape[4]),
								indexing='ij')

		# Grab array
		vavg = cp.mean(cp.array(self.vel), axis=0)
		vx = vavg[0, ...]
		vy = vavg[1, ...]
		vz = vavg[2, ...]


		temp = ( (mag_avg > (mag_thresh * max_mag)) &
					(cd_avg < (cd_thresh * max_cd)) )
		mask = cp.zeros_like(temp)
		ss = 2 #subsample
		mask[::ss,::ss,::ss] = temp[::ss,::ss,::ss]

		# Subselect values
		idx = cp.argwhere(mask)
		x_slice = x[idx[:,0],idx[:,1],idx[:,2]]
		y_slice = y[idx[:,0],idx[:,1],idx[:,2]]
		z_slice = z[idx[:,0],idx[:,1],idx[:,2]]
		vx_slice = vx[idx[:,0],idx[:,1],idx[:,2]]
		vy_slice = vy[idx[:,0],idx[:,1],idx[:,2]]
		vz_slice = vz[idx[:,0],idx[:,1],idx[:,2]]

		for ii in range(N):
			for jj in range(N):
				AhA[ii, jj] = cp.sum( (x_slice ** px[ii] * y_slice ** py[ii] * z_slice ** pz[ii]) *
										(x_slice ** px[jj] * y_slice ** py[jj] * z_slice ** pz[jj]) )
				
		for ii in range(N):
			phi = cp.power(x_slice, px[ii]) * cp.power(y_slice, py[ii]) * cp.power( z_slice, pz[ii])
			AhBx[ii] = cp.sum(vx_slice * phi)
			AhBy[ii] = cp.sum(vy_slice * phi)
			AhBz[ii] = cp.sum(vz_slice * phi)

		polyfit_x = cp.linalg.solve(AhA, AhBx)
		polyfit_y = cp.linalg.solve(AhA, AhBy)
		polyfit_z = cp.linalg.solve(AhA, AhBz)

		background_phase = cp.zeros_like(vx)[None, ...]
		background_phase = cp.concatenate([background_phase, background_phase, background_phase], 0)


		for ii in range(N):
			phi = (x**px[ii])
			phi*= (y**py[ii])
			phi*= (z**pz[ii])
			background_phase[0, ...] += polyfit_x[ii]*phi
			background_phase[1, ...] += polyfit_y[ii]*phi
			background_phase[2, ...] += polyfit_z[ii]*phi


		#Expand and subtract)
		background_phase = background_phase[None, ...]
		background_phase = cp.concatenate([background_phase]*self.vel.shape[0])
		vel_temp = cp.array(self.vel)
		vel_temp -= background_phase
		self.vel = vel_temp.get()
		self.vel_mag = np.sqrt(self.vel[:,0,...]**2 + self.vel[:,1,...]**2 + self.vel[:,2,...]**2)

		# self.vel -= (background_phase.get())
	

if __name__ == "__main__":


	base_path = '//media/buntess/OtherSwifty/Data/ITGADO/gait3/reconed_framed10_wexp0.75_0.010000_29.h5'

	venc = 1100

	print(f'Loading Image')
	with h5py.File(base_path, 'r') as hf:
		image = np.array(hf['image'])
		print(image.shape)

	image = image.reshape(10, 5, 320, 320, 320)
		
	post4DFlow = PostP_4DFlow(venc, image)
	post4DFlow.solve_velocity()
	post4DFlow.update_cd()
	ort.image_nd(post4DFlow.mag) #vel[:,2,...])
		
	print(1)
	# import load_data
	# smaps, image = load_data.load_smaps_image('/media/buntess/OtherSwifty/Data/PET-MR/Wahlin_Test/reconed_iSENSE_2.h5')
	# ort.image_nd(image)
	# image = np.tile(image, (10, 1, 1, 1))
	# image = image.reshape(10, 5, 256, 256, 256)
	# post4DFlow.update_mag(image, 1)

	post4DFlow.correct_background_phase()
	post4DFlow.update_cd()
	ort.image_nd(post4DFlow.cd) #vel[:,2,...])

	filename = '/media/buntess/OtherSwifty/Data/Garpen/Ena/garpen_framed_postp.h5'
	#filename = '/media/buntess/OtherSwifty/Data/Garpen/Ena/garpen_notframed_postp.h5'
	with h5py.File(filename, 'w') as f:
		f.create_dataset('vel', data=post4DFlow.vel)
		f.create_dataset('cd', data=post4DFlow.cd)



	print(1)