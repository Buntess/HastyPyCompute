import math
import cupy as cp
import numpy as np
import h5py

async def fista(xp, x, alpha, gradstep, prox, maxiter, saveImage = False, fileName = None):

    t = xp.array([1.0])
    resids = []

    x_old = xp.empty_like(x)
    z = xp.empty_like(x)
    xp.copyto(z, x)

    async def update():
        xp.copyto(x_old, x)
        xp.copyto(x, z)

        ret = await gradstep(x, alpha)

        await prox(x, alpha, z)


        t_old = t
        t[:] = 0.5 * (1.0 + math.sqrt(1.0 + 4.0*t_old*t_old))

        xp.subtract(x, x_old, out=z)
        resids.append(xp.linalg.norm(z))
        xp.add(x, ((t_old - 1.0) / t) *z, out=z)

    for i in range(maxiter):
        print(i)
        await update()
        if saveImage:
            if i == 9:
                filename = fileName + '1.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)
            elif i == 19:
                filename = fileName + '2.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)
            elif i == 39:
                filename = fileName + '3.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)
            elif i == 69:
                filename = fileName + '4.h5'
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('image', data=x)
            elif i == 99:
                filename = fileName + '5.h5'
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