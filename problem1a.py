#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:37:46 2021

@author: prannoy
"""

import numpy as np
import cv2
import scipy.fft
import matplotlib.pyplot as plt

# P1(a) identifying the tag using fft
cap = cv2.VideoCapture('Tag1.mp4')

a, frame = cap.read()

gray_or = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray_or, (3, 3), 0)

mask = cv2.inRange(gray, 250, 255)

plt.imshow(mask)
plt.show()
no_bg = cv2.bitwise_and(gray, mask, mask=mask)

trans = scipy.fft.fft2(mask)

fshift = np.fft.fftshift(trans)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

rows = mask.shape[0]
cols = mask.shape[1]
crow, ccol = int(rows / 2), int(cols / 2)
fshift[crow - 100:crow + 100, ccol - 100:ccol + 100] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

img_back = np.uint8(img_back)

edg_img = cv2.Canny(img_back, 100, 200)

contours, hierarchy = cv2.findContours(edg_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

plt.imshow(no_bg, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(np.abs(fshift), cmap='gray')
plt.title('Fourier transform with HPF Image'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(img_back, cmap='gray')
plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(frame, cmap='gray')
plt.title('Image with Contours'), plt.xticks([]), plt.yticks([])
plt.show()
print(hierarchy)

