import numpy as np
import pandas as pd


class HexSegments():
    def __init__(self, hexagon):
        self.hexagon = hexagon.T

    def iter_segments(self):
        points = [(list(point), None) for point in self.hexagon]
        points.append(points[0])
        return points


class Hex():
    def __init__(self, centers, values, hexagon):
        self.centers = centers
        self.values = values
        self.hexagon = hexagon

    def get_array(self):
        return self.values

    def get_offsets(self):
        return self.centers.T.copy()

    def get_paths(self):
        return [HexSegments(self.hexagon)]


def _hexbin(
        x, y,
        C=None,
        reduce_C_function=None,
        gridsize=20,
        aspect_ratio=1,
        mincnt=None,
):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    if C is not None:
        C = np.asanyarray(C)
        p = (x, y, C)
    else:
        p = (x, y)
    p = np.vstack(p)
    mask_notnan = ~np.any(np.isnan(p), axis=0)
    p = p[:, mask_notnan]

    if p.shape[1] == 0:
        return None

    p = p[0:2, :]
    if C is not None:
        C = p[2, :]

    p_min = np.amin(p, axis=1)
    p_max = np.amax(p, axis=1)
    p0 = (p_max + p_min) / 2

    if isinstance(gridsize, (list, tuple)):
        x_gridsize, y_gridsize = gridsize
        y_gridsize *= 2
    else:
        x_gridsize = gridsize
        y_gridsize = int(x_gridsize / aspect_ratio / np.sqrt(3) * 2)

    if x_gridsize % 2 == 0:
        x_gridsize -= 1
    if y_gridsize % 2 == 0:
        y_gridsize -= 1

    resol = np.array([x_gridsize, y_gridsize])
    scale = np.maximum((p_max - p_min), 1) / resol

    e = np.array([[1, 0], [-0.5, np.sqrt(3) / 2]])
    e_scaled = e * scale
    e_scaled_inv = np.linalg.inv(e_scaled).T

    i = (
            (e_scaled_inv[:, :, np.newaxis] * p).sum(axis=1)
            - (e_scaled_inv[:, :] * p0).sum(axis=1)[:, np.newaxis]
    ).round().astype('i4')
    # i = (e_scaled_inv[:, :, np.newaxis] * (p - p0[:, np.newaxis])).sum(axis=1).round().astype('i4')

    idx = i[0, :] * (2 * y_gridsize + 7) + i[1, :] + (y_gridsize + 3)
    idx_unique = np.unique(idx)
    i0_unique = idx_unique // (2 * y_gridsize + 7)
    i1_unique = idx_unique % (2 * y_gridsize + 7) - (y_gridsize + 3)
    i_unique = np.vstack((i0_unique, i1_unique))

    centers = p0[:, np.newaxis] + (i_unique[:, np.newaxis, :] * e_scaled[:, :, np.newaxis]).sum(axis=0)

    _hexagon_x_coords = np.array([1, 0, -1, -1, 0, 1]) * 0.5
    _hexagon_y_coords = np.array([1, 2, 1, -1, -2, -1]) * np.sqrt(3) / 6
    hexagon = np.vstack((_hexagon_x_coords, _hexagon_y_coords))

    values = np.ones(centers.shape[1])

    return centers, values, hexagon * scale[:, np.newaxis]


def hexbin(
        x, y,
        C=None,
        reduce_C_function=None,
        gridsize=20,
        aspect_ratio=1,
        mincnt=None,
):
    centers, values, hexagon = _hexbin(
        x, y,
        C=C,
        reduce_C_function=reduce_C_function,
        gridsize=gridsize,
        aspect_ratio=aspect_ratio,
        mincnt=mincnt
    )
    return Hex(centers, values, hexagon)
