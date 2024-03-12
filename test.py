import h5py
import plot_utility as pu
import orthoslicer as ort
import numpy as np
import image_creation as ic
import matplotlib.pyplot as plt

from sigpy import dcf as dcf
import load_data

def load_orchestra_smaps(path, nCoils = 19, imSize = (320, 320, 320)):
    f = h5py.File(path, 'r')
    smaps = np.empty((nCoils,) + imSize)

    for i in range(nCoils):


        current_Smap = np.array(f['Maps'][f'SenseMaps_{i}'])
        smaps[i, ...] = current_Smap['real'] + 1j*current_Smap['imag']
        
    smaps = np.swapaxes(smaps, 1, 3)
    return smaps

if __name__ == "__main__":


    # Load smaps and full image
    smaps, _ = load_data.load_smaps_image('/media/buntess/OtherSwifty/Data/Garpen/Ena/reconed_lowres_1.h5')
    ort.image_nd(smaps)

    print(1)
