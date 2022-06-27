import cv2
import os
import numpy as np
from PIL import Image
import itertools
from sklearn.feature_extraction import image


def apply_gb(src_img, alpha=4, beta=-4, gamma=128, scale_ratio=20):
    '''
    https://kaggle-forum-message-attachments.storage.googleapis.com/88655/2795/competitionreport.pdf
    '''
    result_img = cv2.addWeighted(src_img, alpha, cv2.GaussianBlur(src_img, (0, 0), scale_ratio), beta, gamma)
    return result_img

def apply_CLAHE(input_img):
    lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def crop_img(img, mask=None, threshold=30):
    """
    This function shall crop out the blank background of the given image and make it looks like a square
    :param
        img: a cv2 opencv-python loaded image
        threshold: int
    :return:
        cropped_image: a cv2 opencv-python loaded image
    """
    assert isinstance(img, np.ndarray), 'incorrect input type: img'
    assert isinstance(mask, np.ndarray), 'incorrect input type: mask'
    assert isinstance(threshold, int), 'incorrect input type: threshold'

    try:
        # img_blur = cv2.blur(img, (3, 3))
        # img_blur = cv2.blur(img_blur, (3, 3))
        # img_blur = img_blur.astype('uint8')
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)

        vertical, horizontal = np.nonzero(threshed_img)
        top = vertical[0]
        bot = vertical[-1]

        threshed_img = np.swapaxes(threshed_img, 0, 1)  # Transpose
        vertical, horizontal = np.nonzero(threshed_img)
        left = vertical[0]
        right = vertical[-1]

        cropped_image = img[top:bot, left:right]
        cropped_mask_img = mask[top:bot, left:right]
        return cropped_image, cropped_mask_img
    except Exception as err_msg:
        print(err_msg)
        return img

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))

    # make sure to always check the size of images first
    # img = img.resize((2048, 2048))

    w, h = img.size
    new_ext = '.png'

    grid = list(itertools.product(range(0, h - h % d, d), range(0, w - w % d, d)))
    for i, j in grid:
        box = (j, i, j + d, i + d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{new_ext}')
        img.crop(box).save(out)

def label2greenclass(shape, label):
    x = np.zeros([shape[0], shape[1]])
    label[label >= 127] = 255
    label[label < 127] = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            # if (label[i, j] == [0, 255, 0]).all() or (label[i, j] == [0, 0, 255]).all():
            if (label[i, j] == [0, 255, 0]).all():
                x[i, j] = 255
            # elif (label[i, j] == [0, 0, 255]).all():
            #     x[i, j] = 255
            else:
                x[i, j] = 0
    return np.uint8(x)

def label2class(shape, label):
    x = np.zeros([shape[0], shape[1]])
    label[label >= 127] = 255
    label[label < 127] = 0
    r = 0
    g = 0
    b = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            # white
            # if (label[i, j] == [255, 255, 255]).all():
            #     x[i, j] = 0
            # black
            if (label[i, j] == [0, 0, 0]).all():
                x[i, j] = 0
            # blue
            elif (label[i, j] == [255, 0, 0]).all():
                x[i, j] = 1
                b = b + 1
            # green
            elif (label[i, j] == [0, 255, 0]).all():
                x[i, j] = 2
                g = g + 1
            # red
            elif (label[i, j] == [0, 0, 255]).all():
                x[i, j] = 3
                r = r + 1
            else:
                x[i, j] = 0
    print('R: ', r, ' G: ', g, ' B: ', b)

    return np.uint8(x)


def class2label(shape, img):
    output_img = np.zeros([shape[0], shape[1], 3])
    # white = np.argwhere(img == 0)
    black = np.argwhere(img == 0)
    red = np.argwhere(img == 1)
    green = np.argwhere(img == 2)
    blue = np.argwhere(img == 3)

    # output_img[tuple(np.transpose(white))] = [255, 255, 255]
    output_img[tuple(np.transpose(black))] = [0, 0, 0]
    output_img[tuple(np.transpose(red))] = [255, 0, 0]
    output_img[tuple(np.transpose(green))] = [0, 255, 0]
    output_img[tuple(np.transpose(blue))] = [0, 0, 255]
    return output_img


def create_path_if_not_exists(path):
    """ If path not exist then create the related dirs in path."""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != exc.errno.EEXIST:
                raise ("create_dir_if_not_exists error")