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