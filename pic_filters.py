import in_out as IO
import model as m
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import gradational_transformations as GT
# from scipy.misc import imresize
from skimage.transform import resize as imresize
import resizing as reS
import analysys
from numba import jit


def averaging_filter(_img, mask_shape=5):
    side = mask_shape // 2
    img = np.empty(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = 0 if i-side < 0 else i - side
            i2 = i + side if i + side < img.shape[0] else img.shape[0]
            j1 = 0 if j-side < 0 else j - side
            j2 = j + side if j + side < img.shape[1] else img.shape[1]
            img[i,j] = _img[i1:i2, j1:j2].mean()
    return img

def median_filter(_img, mask_shape=5):
    side = mask_shape // 2
    img = np.empty(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = 0 if i-side < 0 else i - side
            i2 = i + side+1 if i + side+1 < img.shape[0] else img.shape[0]
            j1 = 0 if j-side < 0 else j - side
            j2 = j + side+1 if j + side+1 < img.shape[1] else img.shape[1]
            img[i,j] = np.median(_img[i1:i2, j1:j2])
    return img

def gradient_sobel(_img, mask="h"):
    """
    :param _img:
    :param mask: h for gorizontal, v for vertical
    :return:
    """
    if mask != "a":
        if mask == "h":
            mask = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])
        else:
            mask = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
        side = len(mask) // 2
        img = np.append([_img[0]], _img, axis=0)
        img = np.append(img, [_img[-1]], axis=0)
        img2 = np.zeros((img.shape[0], img.shape[1]+2))
        img2[:, 1:-1] = img
        img2[:, 1] = img[:,0]
        img2[:, -1] = img[:, -1]
        img = np.zeros(_img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                i1 = i + 1
                j1 = j + 1
                img[i, j] = np.average(img2[i1-side:i1+side+1, j1-side:j1+side+1] * mask)
        return img
    else:

        mask = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
        side = len(mask) // 2
        img = np.append([_img[0]], _img, axis=0)
        img = np.append(img, [_img[-1]], axis=0)
        img2 = np.zeros((img.shape[0], img.shape[1] + 2))
        img2[:, 1:-1] = img
        img2[:, 1] = img[:, 0]
        img2[:, -1] = img[:, -1]
        img = np.zeros(_img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                i1 = i + 1
                j1 = j + 1
                img[i, j] = np.average(img2[i1 - side:i1 + side + 1,
                                       j1 - side:j1 + side + 1] * mask)
        img_1 = img
        mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
        img = np.append([_img[0]], _img, axis=0)
        img = np.append(img, [img[-1]], axis=0)
        img2 = np.zeros((img.shape[0], img.shape[1] + 2))
        img2[:, 1:-1] = img
        img2[:, 1] = img[:, 0]
        img2[:, -1] = img[:, -1]
        img = np.zeros(_img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                i1 = i + 1
                j1 = j + 1
                img[i, j] = np.average(img2[i1 - side:i1 + side + 1,
                                       j1 - side:j1 + side + 1] * mask)
        img_2 = img
        mask = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
        side = len(mask) // 2
        img = np.append([img[0]], img, axis=0)
        img = np.append(img, [img[-1]], axis=0)
        img2 = np.zeros((img.shape[0], img.shape[1] + 2))
        img2[:, 1:-1] = img
        img2[:, 1] = img[:, 0]
        img2[:, -1] = img[:, -1]
        img = np.zeros(_img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                i1 = i + 1
                j1 = j + 1
                img[i, j] = np.average(img2[i1 - side:i1 + side + 1,
                                       j1 - side:j1 + side + 1] * mask)
        img_3 = img
        mask = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])
        side = len(mask) // 2
        img = np.append([img[0]], img, axis=0)
        img = np.append(img, [img[-1]], axis=0)
        img2 = np.zeros((img.shape[0], img.shape[1] + 2))
        img2[:, 1:-1] = img
        img2[:, 1] = img[:, 0]
        img2[:, -1] = img[:, -1]
        img = np.zeros(_img.shape)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                i1 = i + 1
                j1 = j + 1
                img[i, j] = np.average(img2[i1 - side:i1 + side + 1,
                                       j1 - side:j1 + side + 1] * mask)
        img_4 = img
        print(img_1.min(), img_1.max())
        print(img_2.min(), img_2.max())
        print(img_3.min(), img_3.max())
        print(img_4.min(), img_4.max())
        # return m.threshold_filter_2d(img_1,img_1.mean()-img_1.std(),255)+\
        #        m.threshold_filter_2d(img_2,img_2.mean()-img_2.std(),255)+\
        #        m.threshold_filter_2d(img_3,img_3.mean()-img_3.std(),255)+\
        #        m.threshold_filter_2d(img_4,img_4.mean()-img_4.std(),255)
        return m.normalize(np.absolute(img_1))+ \
               m.normalize(np.absolute(img_2))+ \
               m.normalize(np.absolute(img_3))+ \
               m.normalize(np.absolute(img_4))
def laplassian(_img, mask="a"):
    """
    :param _img:
    :param mask: h for gorizontal, v for vertical
    :return:
    """
    if mask == "a":
        mask = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
    else:
        mask = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])
    side = len(mask) // 2
    img = np.append([_img[0]], _img, axis=0)
    img = np.append(img, [_img[-1]], axis=0)
    img2 = np.zeros((img.shape[0], img.shape[1] + 2))
    img2[:, 1:-1] = img
    img2[:, 1] = img[:, 0]
    img2[:, -1] = img[:, -1]
    img = np.zeros(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = i + 1
            j1 = j + 1
            img[i, j] = np.average(img2[i1 - side:i1 + side + 1, j1 - side:j1 + side + 1] * mask)
    return img

def dilatation(_img, mask=3):
    side = mask // 2
    img = np.append([_img[0]] * side, _img, axis=0)
    img = np.append(img, [_img[-1]] * side, axis=0)
    img2 = np.zeros((img.shape[0], img.shape[1] + side*2))
    img2[:, side:-side] = img.copy()
    a = np.array([list(img[:, 0])] * side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, :side] = a
    a = np.array([list(img[:, -1])] * side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, -side:] = a
    img = np.empty(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = i + side
            j1 = j + side
            img[i, j] = img2[i1 - side:i1 + side + 1, j1 - side:j1 + side + 1].any()
    return img

def erosion(_img, mask=3):
    side = mask // 2
    img = np.append([_img[0]]*side, _img, axis=0)
    img = np.append(img, [_img[-1]]*side, axis=0)
    img2 = np.zeros((img.shape[0], img.shape[1] + side*2))
    img2[:, side:-side] = img.copy()
    a = np.array([list(img[:, 0])]*side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, :side] = a
    a = np.array([list(img[:, -1])] * side)
    a = a.reshape((a.shape[1], a.shape[0]))
    img2[:, -side:] = a
    img = np.empty(_img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            i1 = i + side
            j1 = j + side
            img[i, j] = img2[i1 - side:i1 + side + 1, j1 - side:j1 + side + 1].all()
    return img