import numpy as np

def invert(_img, max_val=255):
    return max_val - _img

def gamma_correction(_img,  gamma, A=1):
    return A * _img**gamma

def logarifmic_correction(_img, A=1, base=False):
    if not base:
        return A * np.log(_img + 1)
    else:
        return A * np.log(_img + 1) / np.log(base)