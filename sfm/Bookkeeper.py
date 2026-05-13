import numpy as np

class Bookkeeper:
    """
    Usage:
    bk = Bookkeeper()
    bk.add_view(frame_idx, pose)
    point_idx = bk.add_point(coord3d)
    bk.add_obs(frame_idx, point_idx, coord2d, desc)
    """

    def __init__(self, K, dist_coeffs):
        self.K = K
        self.dist_coeffs = dist_coeffs

        self.Tvws = {} # views
        self.Tpts = [] # 3d points
        self.Tobs = {} # observations

        self.Tdcs = {} # descriptors

    def __repr__(self):
        return f"Bookkeeper(Views: {len(self.Tvws)}, Points: {len(self.Tpts)}, Obs: {len(self.Tobs)})"

    def add_view(self, frame_idx, pose):
        self.Tvws[frame_idx] = pose

    def add_point(self, coord):
        self.Tpts.append(coord)
        point_idx = len(self.Tpts) - 1
        self.Tdcs[point_idx] = []
        return point_idx

    def add_obs(self, frame_idx, point_idx, coord, desc):
        self.Tobs[(frame_idx, point_idx)] = coord
        self.Tdcs[point_idx].append(desc)

    def get_view(self, frame_idx):
        return self.Tvws.get(frame_idx)

    def get_point(self, point_idx):
        return self.Tpts[point_idx]

    def get_obs(self, frame_idx, point_idx):
        return self.Tobs.get((frame_idx, point_idx))

    def get_dcs(self):
        point_indices = []
        point_descs = []

        for point_idx, point_desc in self.Tdcs.items():
            for desc in point_desc:
                point_indices.append(point_idx)
                point_descs.append(desc)

        # if not point_descs:
        #     return np.empty((0, 128), dtype=np.float32), np.empty(0, dtype=int)

        return np.asarray(point_indices, dtype=int), np.asarray(point_descs, dtype=np.float32)
