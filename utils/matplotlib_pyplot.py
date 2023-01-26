import numpy as np


class HexSegments():
    def __init__(self):
        pass

    def iter_segments(self):
        return [
            ([-4, -4], None),
            ([-4, 4], None),
            ([4, 4], None),
            ([4, -4], None),
            ([-4, -4], None),
        ]


class Hex():
    def __init__(self):
        pass

    def get_array(self):
        return np.array([100, 200])

    def get_offsets(self):
        return np.array([[10, 20], [20, 10]])

    def get_paths(self):
        return [HexSegments()]


def hexbin(*args, **kwargs):
    return Hex()
