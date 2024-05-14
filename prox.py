import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy as cpxsp
import sigpy.wavelet as wavelet

import plot_utility as pu
import concurrent
import asyncio

import svt

def dctprox(base_alpha):

    async def dctprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        cp.fuse(kernel_name='softmax')
        def softmax(img, lam):
            return cp.exp(1j*cp.angle(img)) * cp.maximum(0, (cp.abs(img) - lam))

        for i in range(image.shape[0]):
            gpuimg = cpxsp.fft.dctn(cp.array(image[i,...]))
            gpuimg = softmax(gpuimg, lamda)
            gpuimg = cpxsp.fft.idctn(gpuimg)
            if hasattr(image, 'device'):
                if image.device == gpuimg.device:
                    cp.copyto(image[i,...], gpuimg)
            else:
                image[i,...] = gpuimg.get()

    return dctprox_ret

def fftprox(base_alpha):

    async def fftprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        def softmax(img, lam):
            return cp.exp(1j*cp.angle(img)) * cp.maximum(0, (cp.abs(img) - lam))
        
        for i in range(image.shape[0]):
            gpuimg = cp.fft.fftn(cp.array(image[i,...]), norm="ortho")
            gpuimg = softmax(gpuimg, lamda)
            gpuimg = cp.fft.ifftn(gpuimg, norm="ortho")
            if hasattr(image, 'device'):
                if image.device == gpuimg.device:
                    cp.copyto(image[i,...], gpuimg)
            else:
                image[i,...] = gpuimg.get()

    return fftprox_ret

def timefftprox(base_alpha):

    async def timefftprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        def softmax(img, lam):
            return cp.exp(1j*cp.angle(img)) * cp.maximum(0, (cp.abs(img) - lam))

        scratchmem.fill(0.0)

        scratchmem = image.reshape((image.shape[0] // 5, 5) + image.shape[1:])

        for en in range(5):
            for fi in range(image.shape[2]):
                subpart = cp.array(scratchmem[:, en, fi, ...])

                subpart = cp.fft.fft(subpart, axis=0, norm='ortho')

                subpart = softmax(subpart, lamda)

                subpart = cp.fft.ifft(subpart, axis=0, norm='ortho')

                scratchmem[:, en, fi, ...] = subpart.get()

                print(fi)




        # scratchmem = np.fft.fft(scratchmem, axis=0, norm='ortho')

        # scratchmem = softmax(scratchmem, lamda)

        # scratchmem = np.fft.ifft(scratchmem, axis=0, norm='ortho')

        np.copyto(image, scratchmem.reshape(image.shape))

    return timefftprox_ret



def svtprox(base_alpha, blk_shape, blk_strides, block_iter):

    async def svtprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        scratchmem.fill(0.0)
        await svt.my_svt3(scratchmem, image,  lamda, blk_shape, blk_strides, block_iter, 5)
        #svt.svt_numba3(scratchmem, image,  lamda, blk_shape, blk_strides, block_iter, 5)

        np.copyto(image, scratchmem)

    return svtprox_ret


def spatial_svtprox(base_alpha, blk_shape, blk_strides, block_iter):

    async def spatial_svtprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        scratchmem.fill(0.0)
        await svt.my_spatial_svt3(scratchmem, image, lamda, blk_shape, blk_strides, block_iter)

        np.copyto(image, scratchmem)

    return spatial_svtprox_ret



def waveletprox(base_alpha):

    async def waveletprox_ret(image, alpha, scratchmem):

        lamda = base_alpha * alpha

        def softmax(img, lam):
            return np.exp(1j*np.angle(img)) * np.maximum(0, (np.abs(img) - lam))
        
        iShape = image[0,...].shape
        
        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        
        futures = []

        def onerun(img):
            gpuimg = wavelet.fwt(img)
            gpuimg = softmax(gpuimg, lamda)
            return wavelet.iwt(gpuimg, iShape)

        print('Starting wavelets')
        for i in range(image.shape[0]):
            futures.append(loop.run_in_executor(executor, 
                onerun, image[i,...]))
        
        print('Begin to wait on Wavelets')
        for i, fut in enumerate(futures):
            image[i,...] = await futures[i]
        print('Waited on wavelets')
        

    return waveletprox_ret


def average_waveletLLR_prox(base_alpha_wave, base_alpha_LLR, blk_shape, blk_strides, block_iter):

    llrprox = svtprox(base_alpha_LLR, blk_shape, blk_strides, block_iter)
    waveprox = waveletprox(base_alpha_wave)

    async def waveletprox_ret(image, alpha, scratchmem):

        image_c = image.copy()

        await waveprox(image_c, alpha, scratchmem)
        await llrprox(image, alpha, scratchmem)

        image = 0.5*(image_c + image)


    return waveletprox_ret




def average_DCT_LLR_prox(base_alpha_dct, base_alpha_LLR, blk_shape, blk_strides, block_iter):

    llrprox = svtprox(base_alpha_LLR, blk_shape, blk_strides, block_iter)
    dct_prox = dctprox(base_alpha_dct)

    async def dctprox_ret(image, alpha, scratchmem):

        image_c = image.copy()

        await dct_prox(image_c, alpha, scratchmem)
        await llrprox(image, alpha, scratchmem)

        image = 0.5*(image_c + image)


    return dctprox_ret