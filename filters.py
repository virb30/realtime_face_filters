import cv2
import numpy as np


def grayscale(image):
    converted_image = np.array(image)
    gray_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2GRAY)
    return gray_image


def sketch(image):
    converted_image = np.array(image)
    gray_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2GRAY)
    inv_gray_image = 255 - gray_image
    blur_image = cv2.GaussianBlur(inv_gray_image, (21, 21), 0, 0)
    sketch_image = cv2.divide(gray_image, 255 - blur_image, scale=256)
    return sketch_image


def sepia(image):
    converted_image = np.array(image)
    converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
    kernel = np.array([[0.272, 0.534, 0.132],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.filter2D(converted_image, -1, kernel)
    sepia_image = cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB)
    return sepia_image


def blur(image):
    b_amount = 9
    converted_image = np.array(image)
    converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
    blur_image = cv2.GaussianBlur(converted_image, (b_amount, b_amount), 0, 0)
    blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
    return blur_image


def canny(image):
    threshold1 = 100
    threshold2 = 150
    converted_image = np.array(image)
    converted_image = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
    blur_image = cv2.GaussianBlur(converted_image, (11, 11), 0)
    canny_image = cv2.Canny(blur_image, threshold1, threshold2)
    return canny_image


def original(image):
    return image


