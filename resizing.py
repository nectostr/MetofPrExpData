import numpy as np
from numba import jit
from numba import generated_jit

@jit
def resize_next_neigbour(_img, mul):
    img = np.zeros(np.array(np.around(np.array(_img.shape) * mul),dtype=np.int))
    mul_ = int(mul)
    for i in range(img.shape[0] - mul_):
        for j in range(img.shape[1] - mul_):
            img[i,j] = _img[round(i/mul) , round(j/mul)]
    for i in range(img.shape[0] - mul_, img.shape[0]):
        for j in range(img.shape[1] - mul_, img.shape[1]):
            img[i, j] = _img[int(i // mul), int(j // mul)]
    return img

def resize_next_neigbour2(_img, mul):
    img = np.zeros(np.array(np.around(np.array(_img.shape) * mul),dtype=np.int))
    for i in range(img.shape[0]-mul):
        for j in range(img.shape[1] - mul):
            img[i,j] = _img[round(i/mul) , round(j/mul)]
    return img


def resize_bilinear_interpolation2(_img, mul):
    # img = np.zeros((round(_img.shape[0]*mul),
    #                round(_img.shape[1]*mul)))
    img = np.zeros(np.array(np.around(np.array(_img.shape) * mul), dtype=np.int))
    for x in range(1, img.shape[0]):
        for y in range(1, img.shape[1]):
            x1 = int(x//mul) - 1
            x2 = x1 + 1
            y1 = int(y//mul) - 1
            y2 = y1 + 1
            x_ = x / mul
            y_ = y / mul
            img[x,y] = _img[x1, y1] * (x2 - x_) * (y2 - y_) +\
            _img[x2, y1]  * (x_ - x1) * (y2 - y_) +\
            _img[x1, y2] * (x2 - x_) * (y_ - y1) +\
            _img[x2, y2]  * (x_ - x1) * (y_ - y1)
    return img

def resize_bilinear_interpolation(_img, mul):
    img = np.zeros(np.array(np.floor(np.array(_img.shape) * mul), dtype=np.int))
    for x in range(1,img.shape[0]-1):
        for y in range(1,img.shape[1]-1):
            x1 = int(x//mul)
            x2 = x1 + 1
            y1 = int(y//mul)
            y2 = y1 + 1
            x_ = x / mul
            y_ = y / mul
            img[x,y] = _img[x1, y1] * (x2 - x_) * (y2 - y_) +\
                       _img[x2, y1] * (x_ - x1) * (y2 - y_) +\
                       _img[x1, y2] * (x2 - x_) * (y_ - y1) +\
                       _img[x2, y2] * (x_ - x1) * (y_ - y1)
    return img