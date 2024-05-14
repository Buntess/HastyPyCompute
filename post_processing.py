import cupy as cp
import h5py
import numpy as np
import math

import plot_utility as pu
import orthoslicer as ort

import gc


def constant_from_venc(venc):
	return (np.pi / (venc * math.sqrt(3)))

def get_encode_matrix(k):
	Emat = k * np.array(
		[
			#[ 0,  0,  0],
			[-1, -1, -1],
			[ 1,  1, -1],
			[ 1, -1,  1],
			[-1,  1,  1]
		], dtype=np.float32)
	return Emat

def _enc_to_vel_linear(image, venc, nwraps):
	Emat = get_encode_matrix(constant_from_venc(venc))
	pEmat = np.linalg.pinv(Emat)

	im_size = (image.shape[1], image.shape[2], image.shape[3])
	nvoxel = math.prod(list(im_size))

	base_phase = np.angle(image[0,...])
	base_corrected = (image * np.exp(-1j*base_phase)[None, ...]).reshape((5,nvoxel))

	phases = np.angle(base_corrected) + 2 * np.pi * nwraps.reshape((5,nvoxel))

	imageout = pEmat @ phases[1:,...]
	#print(imageout.min())
	#print(imageout.max())
	#print(venc * np.sqrt(3))

	mag = np.mean(np.abs(base_corrected), axis=0)[None,...]

	imageout = np.concatenate([mag, imageout], axis=0)

	return (imageout, base_corrected)

def enc_to_vel_linear(images, venc, nwraps):

	plotangle = False
	
	if plotangle:
		ort.image_nd(np.angle(images[:,0,...]))
		ort.image_nd(np.angle(images[:,1,...]))
		ort.image_nd(np.angle(images[:,2,...]))
		ort.image_nd(np.angle(images[:,3,...]))
		ort.image_nd(np.angle(images[:,4,...]))

	images_out = np.empty((images.shape[0], 4, images.shape[2], images.shape[3], images.shape[4]))
	for i in range(images.shape[0]):
		print(f'\r Frame: {i}')
		image = images[i,...]
		images_out[i,...] = _enc_to_vel_linear(np.array(image), venc, np.array(nwraps[i, ...]))[0].reshape((4,images.shape[2], 
								  images.shape[3], images.shape[4]))
	if plotangle:
		ort.image_nd(images_out[:,0,...])
		ort.image_nd(images_out[:,1,...])
		ort.image_nd(images_out[:,2,...])
		ort.image_nd(images_out[:,3,...])

	return images_out




class PostP_4DFlow:

	def __init__(self, venc, image):
		#'Initialization'
		self.venc = venc  #m/s
		self.nwraps = np.zeros_like(np.real(image))
		
		# Matrices
		self.vel = None
		self.cd = None
		self.mag = np.mean(np.abs(image), axis=1)
		self.image = image


	def solve_velocity(self):
		temp = enc_to_vel_linear(np.array(self.image), self.venc, np.array(self.nwraps))
		self.vel = temp[:, 1:, ...]
	
	def update_cd(self):

		vmag = np.sqrt(np.mean(np.abs(np.array(self.vel))**2 , 1))
		self.cd = (np.array(self.mag)*np.sin(math.pi/2.0*vmag/self.venc))

		idx = np.where(vmag > (self.venc/2))
		mag_temp = np.array(self.mag)
		cd_temp = np.array(self.cd)
		cd_temp[idx] = mag_temp[idx]
		self.cd = cd_temp

	def correct_background_phase(self, mag_thresh=0.08, cd_thresh=0.3, fit_order=2):
		
		# Average and max
		mag_avg = np.mean(np.array(self.mag), 0)
		cd_avg = np.mean(np.array(self.cd), 0)

		max_mag = np.max(mag_avg)
		max_cd = np.max(cd_avg)

		# Get the number of coeficients
		range_temp = np.arange(0, fit_order+1)
		pz,py,px = np.meshgrid(range_temp, range_temp, range_temp)
		idx = np.where( (px+py+pz) <= fit_order )
		px = px[idx]
		py = py[idx]
		pz = pz[idx]
		N = len(px)

		AhA = np.zeros([N, N], dtype=np.float32)
		AhBx = np.zeros([N, 1], dtype=np.float32)
		AhBy = np.zeros([N, 1], dtype=np.float32)
		AhBz = np.zeros([N, 1], dtype=np.float32)


		# Now gather terms (Nt x Nz x Ny x Nx x 3 )
		z, y, x = np.meshgrid(np.linspace(-1, 1, self.vel.shape[2]),
								np.linspace(-1, 1, self.vel.shape[3]),
								np.linspace(-1, 1, self.vel.shape[4]),
								indexing='ij')

		# Grab array
		vavg = np.mean(np.array(self.vel), axis=0)
		vx = vavg[0, ...]
		vy = vavg[1, ...]
		vz = vavg[2, ...]


		temp = ( (mag_avg > (mag_thresh * max_mag)) &
					(cd_avg < (cd_thresh * max_cd)) )
		mask = np.zeros_like(temp)
		ss = 2 #subsample
		mask[::ss,::ss,::ss] = temp[::ss,::ss,::ss]

		# Subselect values
		idx = np.argwhere(mask)
		x_slice = x[idx[:,0],idx[:,1],idx[:,2]]
		y_slice = y[idx[:,0],idx[:,1],idx[:,2]]
		z_slice = z[idx[:,0],idx[:,1],idx[:,2]]
		vx_slice = vx[idx[:,0],idx[:,1],idx[:,2]]
		vy_slice = vy[idx[:,0],idx[:,1],idx[:,2]]
		vz_slice = vz[idx[:,0],idx[:,1],idx[:,2]]

		for ii in range(N):
			for jj in range(N):
				AhA[ii, jj] = np.sum( (x_slice ** px[ii] * y_slice ** py[ii] * z_slice ** pz[ii]) *
										(x_slice ** px[jj] * y_slice ** py[jj] * z_slice ** pz[jj]) )
				
		for ii in range(N):
			phi = np.power(x_slice, px[ii]) * np.power(y_slice, py[ii]) * np.power( z_slice, pz[ii])
			AhBx[ii] = np.sum(vx_slice * phi)
			AhBy[ii] = np.sum(vy_slice * phi)
			AhBz[ii] = np.sum(vz_slice * phi)

		polyfit_x = np.linalg.solve(AhA, AhBx)
		polyfit_y = np.linalg.solve(AhA, AhBy)
		polyfit_z = np.linalg.solve(AhA, AhBz)

		background_phase = np.zeros_like(vx)[None, ...]
		background_phase = np.concatenate([background_phase, background_phase, background_phase], 0)


		for ii in range(N):
			phi = (x**px[ii])
			phi*= (y**py[ii])
			phi*= (z**pz[ii])
			background_phase[0, ...] += polyfit_x[ii]*phi
			background_phase[1, ...] += polyfit_y[ii]*phi
			background_phase[2, ...] += polyfit_z[ii]*phi


		#Expand and subtract)
		background_phase = background_phase[None, ...]
		background_phase = np.concatenate([background_phase]*self.vel.shape[0])
		vel_temp = np.array(self.vel)
		vel_temp -= background_phase
		self.vel = vel_temp
		# self.vel -= (background_phase.get())


	

if __name__ == "__main__":

	base_path = '/media/buntess/OtherSwifty/Data/4D2D/long_run/reconed_dct_framed20_blk8_wexp0.50_llr0.00001000_t0.01000000_99.h5'
	venc = 1100

	print(f'Loading Image')
	with h5py.File(base_path, 'r') as hf:
		image = np.array(hf['image'])
		print(image.shape)

	image = image.reshape(20, 5, 256, 256, 256)
		
	post4DFlow = PostP_4DFlow(venc, image)


	#np.save('/media/buntess/OtherSwifty/Data/ITGADO/gait9/gait9.dat', np.transpose(post4DFlow.mag, (1, 2, 3, 0)))
	# with h5py.File('/media/buntess/OtherSwifty/Data/ITGADO/kallaren/mag_time.h5', 'w') as f:
	# 	f.create_dataset('mag1', data=(post4DFlow.mag[0, ...]))
	# 	f.create_dataset('mag2', data=(post4DFlow.mag[1, ...]))
	# 	f.create_dataset('mag3', data=(post4DFlow.mag[2, ...]))
	# 	f.create_dataset('mag4', data=(post4DFlow.mag[3, ...]))
	# 	f.create_dataset('mag5', data=(post4DFlow.mag[4, ...]))
	# 	f.create_dataset('mag6', data=(post4DFlow.mag[5, ...]))
	# 	f.create_dataset('mag7', data=(post4DFlow.mag[6, ...]))
	# 	f.create_dataset('mag8', data=(post4DFlow.mag[7, ...]))
	# 	f.create_dataset('mag9', data=(post4DFlow.mag[8, ...]))
	# 	f.create_dataset('mag10', data=(post4DFlow.mag[9, ...]))


	print('Solve velocity')
	post4DFlow.solve_velocity()
	print('Update CD')
	post4DFlow.update_cd()

	pu.image_nd(post4DFlow.mag[::2] + post4DFlow.mag[1::2]) #vel[:,2,...])

	print('Correct background phase')
	post4DFlow.correct_background_phase(fit_order=3)
	print('Update CD')
	post4DFlow.update_cd()

	# pu.image_nd(post4DFlow.cd) #vel[:,2,...])

	# print('Create Background Corrected Image')
	# Emat = get_encode_matrix(constant_from_venc(venc))
	# vel = np.transpose(post4DFlow.vel, (0,2,3,4,1))
	# phase = Emat @ vel[...,None]
	# corrected_image = np.empty((10, 5, 320, 320, 320), dtype=np.complex64)
	# corrected_image[:,0,...] = post4DFlow.mag
	# corrected_image[:,1:,...] = np.transpose(post4DFlow.mag[...,None] * np.exp(1j*np.squeeze(phase)), (0,4,1,2,3))

	addon_smaps = False
	if addon_smaps:
		print('Loading smaps')
		with h5py.File(base_path + 'smaps_true.h5', 'r') as f:
			smaps = f['smaps'][()]


	print('Write Corrected Images')
	with h5py.File(base_path[:-3] + 'background_corrected_to.h5', 'w') as f:
		f.create_dataset('vel', data=post4DFlow.vel)
		f.create_dataset('cd', data=post4DFlow.cd)
		# f.create_dataset('corrected_img', data=corrected_image)
		# f.create_dataset('smaps', data=smaps)