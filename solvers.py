import math
import cupy as cp
import numpy as np
import h5py
import gc
import time

async def fista(xp, x, alpha, gradstep, prox, maxiter, saveImage = False, fileName = None):

    t = xp.array([1.0])
    resids = []

    x_old = xp.empty_like(x)
    z = xp.empty_like(x)
    xp.copyto(z, x)

    async def update():
        xp.copyto(x_old, x)
        xp.copyto(x, z)

        start = time.time()
        await gradstep(x, alpha)
        end = time.time()
        print(f"Grad Time = {end - start}")

        start = time.time()
        await prox(x, alpha, z)
        end = time.time()
        print(f"Prox Time = {end - start}")

        t_old = t
        t[:] = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*t_old*t_old))

        xp.subtract(x, x_old, out=z)
        resids.append(xp.linalg.norm(z))
        xp.add(x, ((t_old - 1.0) / t) *z, out=z)
        gc.collect()

    for i in range(maxiter):
        print(i)
        await update()
        if saveImage:

            if i == 49:
                filename = fileName + '49.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)
            elif i == 99:
                filename = fileName + '99.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)
            elif i == 199:
                filename = fileName + '199.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)

    return resids

    


async def max_eig(xp, A, x, iter):
    xup = xp.copy(x)
    for i in range(iter):
        y = await A(xup)
        maxeig = xp.linalg.norm(y)
        xup = y / maxeig
    return maxeig