from math import ceil

import cv2
import numpy as np
from PIL import Image

noisyPath = 'res/moonNoisy{}.gif'
originalPath = 'res/moonOriginal.gif'
imageCount = 10


def readGIF(path: str):
    gif = cv2.VideoCapture(path)
    ret, frame = gif.read()
    img = Image.fromarray(frame)
    open_cv_image = np.array(img, dtype='float64')
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image[:, :, 0]


def mean(images: np.ndarray):
    return images.mean(axis=0)


def median(images: np.ndarray):
    return np.median(images, axis=0)


def percentile(images: np.ndarray, percent: int):
    images.sort(axis=0)
    return (images[ceil(percent / images.shape[0]) - 1])


def compare(image1: np.ndarray, image2: np.ndarray):
    # return cv2.PSNR(image1, image2)
    height, width = image1.shape
    differenceImage = image2 - image1
    differenceImage **= 2
    difference = differenceImage.sum().sum()
    PSNR = 10 * np.log10(((255 ** 5) * height * width) / difference)
    return PSNR


images = np.ndarray((imageCount, 538, 464))
for i in range(0, imageCount):
    images[i - 1] = readGIF(noisyPath.format(i + 1))

# noiseReduced = mean(images)
noiseReduced = median(images)
# noiseReduced = percentile(images, 40)

cv2.imwrite('out.png', noiseReduced)

og = readGIF(originalPath)
print('PSNR = {}'.format(compare(noiseReduced, og)))
print('OpenCV PSNR = {}'.format(cv2.PSNR(noiseReduced, og)))
