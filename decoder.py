#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 15:52:48 2021

@author: prannoy
"""

import cv2
import numpy as np

image = cv2.imread('ref_marker.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def decode_image(image):
    image = cv2.resize(image, (200, 200))

    retval = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    image_e = cv2.erode(image,retval)
    
    image_e = cv2.dilate(image_e, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) )

    decoded_from_tag = np.zeros((8, 8))

    row_partitions = list(range(0, 201, 25))
    column_partitions = list(range(0, 201, 25))

    for i in range(0, 8):
        for j in range(0, 8):
            part_of_image = image_e[row_partitions[j]:row_partitions[j + 1], column_partitions[i]:column_partitions[i + 1]]
            mean = np.mean(part_of_image)
            if mean > 30:
                decoded_from_tag[j, i] = 1
    rotations = 0
    while True:
        if decoded_from_tag[5, 5] == 0 and rotations < 4:
            decoded_from_tag = np.rot90(decoded_from_tag)
            rotations += 1
        else:
            break

    direction = {0: "UPRIGHT", 1: "ROTATED BY 90 CW", 2: "UPSIDE DOWN", 3: "ROTATED BY 90 CCW", 4: "ROTATED BY 90 CCW"}

    value = decoded_from_tag[3, 3] * 1 + decoded_from_tag[3, 4] * 2 + decoded_from_tag[4, 4] * 4 + decoded_from_tag[
        4, 3] * 8

    return decoded_from_tag, value, direction[rotations]


info, value, direction = decode_image(gray)




