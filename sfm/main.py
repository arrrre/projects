import argparse
import pickle
from pathlib import Path
import time
import cv2 as cv
import numpy as np
import viser
import viser.transforms as vtf

from utils import find_correspondences, robust_f, E_from_F, R_t_from_E, triangulate, robust_pnp, bundle_adjustment

class Bookkeeper:
    """
    Usage:
    bk = Bookkeeper()
    bk.add_view(frame_idx, pose)
    point_idx = bk.add_point(coord3d)
    bk.add_obs(frame_idx, point_idx, coord2d)
    bk.add_des(point_idx, des)
    """

    def __init__(self):
        self.K = np.array([
            [1660.076384971925, 0, 763.9663384634628],
            [0, 1656.285062933074, 986.3176281647676],
            [0, 0, 1],
        ])
        self.dist_coeffs = np.array([
            0.250022, -1.49939, -0.00296536, 0.00109332, 2.83634
        ])

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
    
    def add_obs(self, frame_idx, point_idx, coord):
        self.Tobs[(frame_idx, point_idx)] = coord

    def add_des(self, point_idx, des):
        self.Tdcs[point_idx].append(des)

    def get_view(self, frame_idx):
        return self.Tvws[frame_idx]
    
    def get_point(self, point_idx):
        return self.Tpts[point_idx]
    
    def get_obs(self, frame_idx, point_idx):
        return self.Tobs[(frame_idx, point_idx)]

def init(bk, img1, img2, mask1, mask2, frame_idx1, frame_idx2):
    corr1, corr2, _, _, des1, des2 = find_correspondences(img1, img2, mask1, mask2, 100, 100)
    n = min(corr1.shape[1], corr2.shape[1]) # TODO: why not the same size already?
    corr1 = corr1[:, :n]; corr2 = corr2[:, :n]

    F, inlier_indices = robust_f(corr1, corr2, N=10000, threshold=1.25)
    corr1 = corr1[:, inlier_indices]
    corr2 = corr2[:, inlier_indices]
    des1 = des1[inlier_indices, :]
    des2 = des2[inlier_indices, :]

    E = E_from_F(F, bk.K, bk.K)
    R, t = R_t_from_E(E, bk.K, bk.K, corr1[:, 0], corr2[:, 0])

    C1 = np.eye(3, 4)
    C2 = np.hstack((R, t))

    bk.add_view(frame_idx1, C1)
    bk.add_view(frame_idx2, C2)

    points_3d = triangulate(C1, C2, bk.K, bk.K, corr1, corr2)
    for i, p_3d in enumerate(points_3d.T):
        p_idx = bk.add_point(p_3d)
        bk.add_obs(frame_idx1, p_idx, corr1[:, i])
        bk.add_obs(frame_idx2, p_idx, corr2[:, i])
        bk.add_des(p_idx, des1[i])
        bk.add_des(p_idx, des2[i])

def loop(bk, img1, img2, mask1, mask2, frame_idx1, frame_idx2):
    corr1, corr2, _, _, des1, des2 = find_correspondences(img1, img2, mask1, mask2, 100, 100)


    

    R, t = robust_pnp()

    C1 = bk.get_view(frame_idx1)
    C2 = np.hstack((R, t))
    bk.add_view(frame_idx2, C2)

    points_3d = triangulate(corr1, corr2, C1, C2, bk.K, bk.K)
    for i, p_3d in enumerate(points_3d.T):
        p_idx = bk.add_point(p_3d)
        bk.add_obs(frame_idx1, p_idx, corr1[:, i])
        bk.add_obs(frame_idx2, p_idx, corr2[:, i])
        bk.add_des(p_idx, des1[i])
        bk.add_des(p_idx, des2[i])

    bundle_adjustment(bk)

def sfm(bk, dataset_path, dataset_name, num_images = 36):
    mask = cv.imread(dataset_path / f"{dataset_name}_mask.png")

    for i in range(1, num_images):
        frame_idx1 = i; frame_idx2 = i + 1
        img1 = cv.imread(dataset_path / f"frame{frame_idx1:02d}.png")
        img2 = cv.imread(dataset_path / f"frame{frame_idx2:02d}.png")

        if i == 1:
            print(f"Initializing with frames {frame_idx1:02d} and {frame_idx2:02d}")
            init(bk, img1, img2, mask, mask, frame_idx1, frame_idx2)
            break
        else:
            print(f"Adding frame {frame_idx2:02d}")
            loop(bk, img1, img2, mask, mask, frame_idx1, frame_idx2)

def visualize(bk, dataset_path):
    print("Starting visualization!")
    now = time.time()
    server = viser.ViserServer()

    points = np.asarray(bk.Tpts, dtype=float).reshape(-1, 3)
    print(f"{points.shape=}")
    if points.shape[0] > 0:
        points_center = np.mean(points, axis=0)
        points = points - points_center
    else:
        points_center = np.zeros(3, dtype=float)

    frame_paths = sorted(dataset_path.glob("frame*.png"))
    frames = [cv.imread(p) for p in frame_paths]
    frame_count = len(frame_paths)

    pixel_colors = []

    for point_idx in range(len(bk.Tpts)):
        for frame_idx in bk.Tvws.keys():
            point_2d = bk.get_obs(frame_idx, point_idx)
            if point_2d is None:
                # this 3D point not visible in this camera!
                continue
            image = frames[frame_idx]
            x, y = point_2d
            b, g, r = image[int(y), int(x)]
            image_pixel_color = [r, g, b]
            # print(f"Color at ({x}, {y}) in camera {frame_idx} for point {point_idx} is {image_pixel_color}")
            pixel_colors.append(image_pixel_color)

            # we found the first image this point appeared in.
            # don't care about rest!
            break

    server.scene.add_point_cloud(
        name="/points",
        points=points,
        colors=np.asarray(pixel_colors),
        point_size=0.02,
        point_shape="circle",
    )

    K = np.asarray(bk.K, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"Expected K as 3x3, got {K.shape}")

    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    width = max(1.0, 2.0 * cx)
    height = max(1.0, 2.0 * cy)
    fov = 2.0 * np.arctan2(height / 2.0, fy)
    aspect = height / width


    if len(bk.Tvws) > 0:
        camera_ups = []
        for C in bk.Tvws.values():
            R = C[:3, :3]
            R = np.asarray(R, dtype=float).reshape(3, 3)
            R_world_camera = R.T
            camera_ups.append(R_world_camera @ np.array([0.0, -1.0, 0.0]))
        average_up = np.mean(np.stack(camera_ups, axis=0), axis=0)
        norm = np.linalg.norm(average_up)
        if norm > 0:
            average_up = average_up / norm
            server.scene.set_up_direction(
                (float(average_up[0]), float(average_up[1]), float(average_up[2]))
            )


    for frame_idx, C in bk.Tvws.items():
        R = np.asarray(C[:3, :3], dtype=float).reshape(3, 3)
        t = np.asarray(C[:3, 3], dtype=float).reshape(3)
        R_world_camera = R.T
        t_world_camera = (-R.T @ t) - points_center

        frame = server.scene.add_frame(
            name=f"/cameras/frame_{frame_idx}",
            wxyz=vtf.SO3.from_matrix(R_world_camera).wxyz,
            position=t_world_camera,
            axes_length=0.08,
            axes_radius=0.004,
        )
        image = None
        if frame_count > 0:
            frame_path = frame_paths[frame_idx % frame_count]
            image = cv.imread(str(frame_path))
        if image is not None:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        server.scene.add_camera_frustum(
            name=f"/cameras/frame_{frame_idx}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.12,
            image=image,
        )

    print(f"Took {time.time() - now:.2f} seconds to set up visualization.")

    while True:
        pass

def main():
    parser = argparse.ArgumentParser(description="SfM")
    parser.add_argument("--dataset_name", type=str, default="my")
    parser.add_argument("--dataset_path", type=str, default="C:/Arvid/code/tsbb33-datasets/")
    parser.add_argument("--use_stored", action='store_true', default=False)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = Path(args.dataset_path) / args.dataset_name
    use_stored = args.use_stored

    if use_stored:
        with open('geometry.pkl', 'rb') as file:
            bk = pickle.load(file)
    else:
        bk = Bookkeeper()
        sfm(bk, dataset_path, dataset_name)

        with open("geometry.pkl", "wb") as file:
            pickle.dump(bk, file)

    print(bk)
    print(bk.Tvws)

    visualize(bk, dataset_path)

if __name__ == '__main__':
    main()
