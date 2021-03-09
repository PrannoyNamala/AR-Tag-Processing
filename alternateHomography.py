import numpy as np


def cubeHomography(corners, projection_corners=None):
    if projection_corners is None:
        projection_corners = {"x": [0, 200, 200, 0], "y": [200, 200, 0, 0]}
    values_dict = {**{"xp": corners[:, 0, 0], "yp": corners[:, 0, 1]}, **projection_corners}

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
