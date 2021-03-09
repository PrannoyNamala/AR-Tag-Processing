# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 15:52:48 2021

@author: prannoy
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from decoder import decode_image
from alternateHomography import cubeHomography


def findHomography(corners, projection_corners=None):
    if projection_corners is None:
        projection_corners = {"xp": [0, 200, 200, 0], "yp": [200, 200, 0, 0]}
    values_dict = {**{"x": corners[:, 0, 0], "y": corners[:, 0, 1]}, **projection_corners}

    A = np.array(([values_dict["x"][0], values_dict["y"][0], 1, 0, 0, 0, -values_dict["x"][0] * values_dict["xp"][0],
                   -values_dict["y"][0] * values_dict["xp"][0], -values_dict["xp"][0]],
                  [0, 0, 0, values_dict["x"][0], values_dict["y"][0], 1, -values_dict["x"][0] * values_dict["yp"][0],
                   -values_dict["y"][0] * values_dict["yp"][0], -values_dict["yp"][0]],
                  [values_dict["x"][1], values_dict["y"][1], 1, 0, 0, 0, -values_dict["x"][1] * values_dict["xp"][1],
                   -values_dict["y"][1] * values_dict["xp"][1], -values_dict["xp"][1]],
                  [0, 0, 0, values_dict["x"][1], values_dict["y"][1], 1, -values_dict["x"][1] * values_dict["yp"][1],
                   -values_dict["y"][1] * values_dict["yp"][1], -values_dict["yp"][1]],
                  [values_dict["x"][2], values_dict["y"][2], 1, 0, 0, 0, -values_dict["x"][2] * values_dict["xp"][2],
                   -values_dict["y"][2] * values_dict["xp"][2], -values_dict["xp"][2]],
                  [0, 0, 0, values_dict["x"][2], values_dict["y"][2], 1, -values_dict["x"][2] * values_dict["yp"][2],
                   -values_dict["y"][2] * values_dict["yp"][2], -values_dict["yp"][2]],
                  [values_dict["x"][3], values_dict["y"][3], 1, 0, 0, 0, -values_dict["x"][3] * values_dict["xp"][3],
                   -values_dict["y"][3] * values_dict["xp"][3], -values_dict["xp"][3]],
                  [0, 0, 0, values_dict["x"][3], values_dict["y"][3], 1, -values_dict["x"][3] * values_dict["yp"][3],
                   -values_dict["y"][3] * values_dict["yp"][3], -values_dict["yp"][3]]))

    u, l, v = np.linalg.svd(A)

    h = v[-1, :].reshape(3, 3)

    h = np.divide(h, h[2, 2])

    return h


def warpPerspective(h, img, shape):
    code_from_video = np.zeros(shape, np.uint8)

    for a in range(0, shape[0]):
        for b in range(0, shape[1]):
            projection = np.matmul(np.linalg.inv(h), [a, b, 1])
            try:
                code_from_video[b, a] = img[int(projection[1] / projection[2]), int(projection[0] / projection[2])]
            except:
                pass

    return code_from_video


def image_projection(input_image, frame, corners, direction):
    direction_key = {"UPRIGHT": 0, "ROTATED BY 90 CW": 1, "UPSIDE DOWN": 2, "ROTATED BY 90 CCW": 3}
    if direction_key[direction] == 0:
        picture_corners = {"xp": [0, 0, 200, 200], "yp": [0, 200, 200, 0]}
    elif direction_key[direction] == 1:
        picture_corners = {"xp": [200, 0, 0, 200], "yp": [0, 0, 200, 200]}
    elif direction_key[direction] == 2:
        picture_corners = {"xp": [200, 200, 0, 0], "yp": [200, 0, 0, 200]}
    elif direction_key[direction] == 3:
        picture_corners = {"xp": [0, 200, 200, 0], "yp": [200, 200, 0, 0]}

    h = findHomography(corners, picture_corners)

    for a in range(0, input_image.shape[0]):
        for b in range(0, input_image.shape[1]):
            projection = np.matmul(np.linalg.inv(h), [a, b, 1])
            try:
                frame[int(projection[1] / projection[2]), int(projection[0] / projection[2]), :] = input_image[b, a, :]
            except:
                pass

    return frame


def cube_projection(corners, frame, direction):
    direction_key = {"UPRIGHT": 0, "ROTATED BY 90 CW": 1, "UPSIDE DOWN": 2, "ROTATED BY 90 CCW": 3}
    if direction_key[direction] == 0:
        picture_corners = {"xp": [0, 0, 200, 200], "yp": [0, 200, 200, 0]}
    elif direction_key[direction] == 1:
        picture_corners = {"xp": [200, 0, 0, 200], "yp": [0, 0, 200, 200]}
    elif direction_key[direction] == 2:
        picture_corners = {"xp": [200, 200, 0, 0], "yp": [200, 0, 0, 200]}
    elif direction_key[direction] == 3:
        picture_corners = {"xp": [0, 200, 200, 0], "yp": [200, 200, 0, 0]}

    h = findHomography(corners, picture_corners)

    h = np.linalg.inv(h)

    h = cubeHomography(corners)

    k = np.array(((1406.08415449821, 2.206797873085990, 1014.13643417416), (0, 1417.99930662800, 566.347754321696),
                  (0, 0, 1)))

    lambda_ = 2 / (
        (np.linalg.norm(np.matmul(np.linalg.inv(k), (h[:, 0]))) + np.linalg.norm(np.matmul(np.linalg.inv(k), h[:, 1]))))

    b_tilda = lambda_ * np.matmul(np.linalg.inv(k), h)

    b = (lambda_ * b_tilda)

    det = np.linalg.det(b_tilda)

    if det < 0:
        b *= -1
        det = np.linalg.det(b)

    r1 = b[:, 0]

    r2 = b[:, 1]

    r3 = np.cross(r1, r2) / lambda_

    t = b[:, 2]

    R = np.column_stack((r1, r2, r3, t))

    p = np.matmul(k, R)

    cube_points = {1: (0, 0, 0, 1), 2: (200, 0, 0, 1), 3: (200, 200, 0, 1), 4: (0, 200, 0, 1), 5: (200, 200, -200, 1),
                   6: (200, 0, -200, 1),
                   7: (0, 0, -200, 1),
                   8: (0, 200, -200, 1)}

    projection_points = {}

    for i in range(1, 9):
        x, y, s = np.matmul(p, np.array(cube_points[i]))
        projection_points[i] = (int(x / s), int(y / s))

    line_pairs = [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (5, 8), (1, 7), (2, 6), (3, 5), (4, 8)]

    for (i, j) in line_pairs:
        cv2.line(frame, projection_points[i], projection_points[j], (255, 0, 0), 1)

    return frame


def execution1b(frame):
    gray_or = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray_or, (3, 3), 0)

    mask = cv2.inRange(gray, 250, 255)

    no_bg = cv2.bitwise_and(gray, mask, mask=mask)

    contours, hierarchy = cv2.findContours(no_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tag_contours_indices = []

    for info in hierarchy[0]:
        if info[3] == -1:
            tag_contours_indices.append(int(info[2]))

    tag_contours = []

    for i in range(len(contours)):
        if i in tag_contours_indices:
            tag_contours.append(contours[i])

    corners = []

    for contour in tag_contours:
        perimeter = cv2.arcLength(contour, True)
        corner_set = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        corners.append(corner_set)

    input_image = cv2.imread('testudo.png')
    input_image = cv2.resize(input_image, (200, 200))

    for corner_set in corners:
        try:
            projected_image = warpPerspective(findHomography(corner_set), mask, (200, 200))
            info, value, direction = (decode_image(projected_image))
            frame = image_projection(input_image, frame, corner_set, direction)
            frame = cube_projection(corner_set, frame, direction)
        except:
            continue
    return frame
