import cv2
from PIL import Image
import numpy as np


def save_from_array(imgs, path):
    i = 0
    for img in imgs:
        cv2.imwrite(path + '/{}.jpg'.format(str(i)), img[:, :, ::-1])
        i += 1

def check_size(dir,shape, length, format='.png'):

    for i in range(length):
        name = dir + '/{}'.format(str(i)) + format
        img = Image.open(name)
        img = np.array(img)

        if img.shape[0] != shape[0]:
            img = cv2.resize(img,shape)
            cv2.imwrite(name, img)

def add_texture(img, texture, channel_weight, threshold=200):
    """Enhanced by green-sreen technology.

    Args:
        img (ndarray): input image.
        texture (ndarray): texture img. img and texture must have the same shape.
        channel_weight (ndarray): The color of target object.
    """

    weight_img = img[:,:,0]*channel_weight[0]+img[:,:,1]*channel_weight[1]+img[:,:,2]*channel_weight[2]

    index = np.where(weight_img>threshold)

    img[index[0],index[1],:] = texture[index[0],index[1],:]

    return img

def gray_img(img, channel_weight):
    weight_img = img[:,:,0]*channel_weight[0]+img[:,:,1]*channel_weight[1]+img[:,:,2]*channel_weight[2]
    return weight_img

