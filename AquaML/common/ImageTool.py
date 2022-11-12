import cv2


def save_from_array(imgs, path):
    i = 0
    for img in imgs:
        cv2.imwrite(path + '/{}.jpg'.format(str(i)), img[:, :, ::-1])

    i += 1
