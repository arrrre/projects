import argparse
import pickle
from pathlib import Path
import time
import cv2 as cv
import numpy as np
import viser
import viser.transforms as vtf

from Bookkeeper import Bookkeeper

from utils2 import robust_f, fmatrix_from_cameras, E_from_F, R_t_from_E, triangulate, robust_pnp
from utils.bundle_adjustment import bundle_adjustment
from utils.correspondence import interest_points, match_points, find_2d_correspondences, find_2d_3d_correspondences

def init(bk, frame_idx1, frame_idx2, img1, img2, mask1, mask2):
    """
    Initialization of incremental bundle adjustment.
    According to algorithm 21.1 in IREG.
    """
    kp1, kp2, desc1, desc2, _, _ = find_2d_correspondences(img1, img2, mask1, mask2, 90, 90)
    corr1 = np.asarray([kp.pt for kp in kp1]).T
    corr2 = np.asarray([kp.pt for kp in kp2]).T

    F, inlier_indices = robust_f(corr1, corr2, N=10000, threshold=1.25)
    corr1 = corr1[:, inlier_indices]
    corr2 = corr2[:, inlier_indices]
    desc1 = desc1[inlier_indices, :]
    desc2 = desc2[inlier_indices, :]

    E = E_from_F(F, bk.K, bk.K)
    R, t = R_t_from_E(E, bk.K, bk.K, bk.dist_coeffs, bk.dist_coeffs, corr1[:, 0], corr2[:, 0])

    C1 = np.eye(3, 4)
    C2 = np.hstack((R, t))

    bk.add_view(frame_idx1, C1)
    bk.add_view(frame_idx2, C2)

    points_3d = triangulate(C1, C2, bk.K, bk.K, bk.dist_coeffs, bk.dist_coeffs, corr1, corr2)
    for i, p_3d in enumerate(points_3d.T):
        p_idx = bk.add_point(p_3d)
        bk.add_obs(frame_idx1, p_idx, corr1[:, i], desc1[i])
        bk.add_obs(frame_idx2, p_idx, corr2[:, i], desc2[i])

def adjust(bk, points_to_adjust):
    """
    Select inliers: T'points
    Bundle adjustment
    Re-triangulate
    Remove outliers
    """
    bundle_adjustment(bk)

def add_new_view(bk, frame_idx1, frame_idx2, img1, img2, mask1, mask2):
    """
    Add new view to the SfM problem according to algorithm 21.2 in IREG.
    """
    # 1. Standard Feature Extraction
    kp1, desc1 = interest_points(img1, mask1, n_percent=100)
    kp2, desc2 = interest_points(img2, mask2, n_percent=100)

    # 2. Handle PnP (Localization)
    indices_3d, descs_3d = bk.get_dcs()
    pt_indices_3d, pt_indices_2d = find_2d_3d_correspondences(indices_3d, descs_3d, desc2)

    if len(pt_indices_3d) < 6:
        raise ValueError(f"PnP failed: only {len(pt_indices_3d)} 2D-3D matches found.")

    corr2d_pnp = np.array([kp2[i].pt for i in pt_indices_2d]).T
    corr3d_pnp = np.array([bk.Tpts[i] for i in pt_indices_3d]).T

    print(f"{corr2d_pnp.shape=}, {corr3d_pnp.shape=}")

    R2, t2, inlier_indices = robust_pnp(corr2d_pnp, corr3d_pnp, bk.K, bk.dist_coeffs)
    bk.add_view(frame_idx2, np.hstack((R2, t2)))

    # Record observations for existing 3D points
    for i in inlier_indices:
        p_3d_idx = pt_indices_3d[i]
        p_2d_idx = pt_indices_2d[i]
        bk.add_obs(frame_idx2, p_3d_idx, kp2[p_2d_idx].pt, desc2[p_2d_idx])


    # Match kp1 and kp2 to find potential new 3D points
    # kp1m/kp2m are the matched keypoints, desc1m/desc2m are their descriptors
    # Make sure your match_points returns the indices as well!
    kp1m, kp2m, desc1m, desc2m, idx1, idx2 = match_points(kp1, kp2, desc1, desc2, n_percent=100)

    # Filter out points that are ALREADY in the 3D library
    # We don't want to duplicate points
    pnp_used_2d_indices = set(pt_indices_2d[inlier_indices])
    
    mask_new = [i not in pnp_used_2d_indices for i in idx2]
    new_pts1 = np.array([p.pt for p in np.array(kp1m)[mask_new]]).T
    new_pts2 = np.array([p.pt for p in np.array(kp2m)[mask_new]]).T
    new_descs1 = np.array(desc1m)[mask_new]
    new_descs2 = np.array(desc2m)[mask_new]

    # return new_pts1, new_pts2, new_descs1, new_descs2

    if len(new_pts1.T) > 0:
        # Get Poses
        C1 = bk.get_view(frame_idx1) # [R1 | t1]
        C2 = bk.get_view(frame_idx2) # [R2 | t2]
        R1 = C1[:3, :3]; t1 = C1[:, 3]
        R2 = C2[:3, :3]; t2 = C2[:, 3]

        X_new = triangulate(C1, C2, bk.K, bk.K, bk.dist_coeffs, bk.dist_coeffs, new_pts1, new_pts2)

        for i in range(X_new.shape[1]):
            p_3d = X_new[:, i]

            # Project back to both images

            p1_proj, _ = cv.projectPoints(p_3d, R1, t1, bk.K, bk.dist_coeffs)
            p2_proj, _ = cv.projectPoints(p_3d, R2, t2, bk.K, bk.dist_coeffs)

            err1 = np.linalg.norm(p1_proj - new_pts1[:, i])
            err2 = np.linalg.norm(p2_proj - new_pts2[:, i])

            if err1 < 5.0 and err2 < 5.0: 
                p_idx = bk.add_point(p_3d)
                bk.add_obs(frame_idx1, p_idx, new_pts1[:, i], new_descs1[i])
                bk.add_obs(frame_idx2, p_idx, new_pts2[:, i], new_descs2[i])

def add_new_3d_points(bk, frame_idx1, frame_idx2, corr1, corr2, desc1, desc2, threshold = 5.0):
    """
    Add new 3D points to the SfM problem according to algorithm 21.3 in IREG.
    """
    C1 = bk.get_view(frame_idx1)
    C2 = bk.get_view(frame_idx2)
    F = fmatrix_from_cameras(C1, C2)
    E = E_from_F(F, bk.K, bk.K)
    R2 = C2[:3, :3]; t2 = C2[:, 3]

    points_3d = triangulate(C1, C2, bk.K, bk.K, corr1, corr2)
    # points_proj = cv.projectPoints(points_3d.T, R2, t2, bk.K, bk.dist_coeffs)
    for i, p_3d in enumerate(points_3d.T):
        point_proj, _ = cv.projectPoints(p_3d, R2, t2, bk.K, bk.dist_coeffs)
        point_proj = point_proj.squeeze()

        error = np.linalg.norm(point_proj - corr2[:, i])
        if error < threshold:
            p_idx = bk.add_point(p_3d)
            bk.add_obs(frame_idx1, p_idx, corr1[:, i], desc1[i])
            bk.add_obs(frame_idx2, p_idx, corr2[:, i], desc2[i])

def sfm(bk, dataset_path, dataset_name, num_images = 36):
    mask = cv.imread(dataset_path / f"{dataset_name}_mask.png")

    for i in range(1, num_images):
        frame_idx1 = i; frame_idx2 = i + 1
        img1 = cv.imread(dataset_path / f"frame{frame_idx1:02d}.png")
        img2 = cv.imread(dataset_path / f"frame{frame_idx2:02d}.png")

        if i == 1:
            print(f"Initializing with frames {frame_idx1:02d} and {frame_idx2:02d}")
            init(bk, frame_idx1, frame_idx2, img1, img2, mask, mask)
            points_to_adjust = len(bk.Tpts)
        else:
            print(f"Adding frame {frame_idx2:02d}")
            # loop(bk, img1, img2, mask, mask, frame_idx1, frame_idx2)
            # corr1_no3d, corr2_no3d, desc1_no3d, desc2_no3d = add_new_view(
            #     bk, frame_idx1, frame_idx2, img1, img2, mask, mask
            # )
            points_to_adjust = len(bk.Tpts)
            # add_new_3d_points(bk, frame_idx1, frame_idx2, corr1_no3d, corr2_no3d, desc1_no3d, desc2_no3d)

            add_new_view(bk, frame_idx1, frame_idx2, img1, img2, mask, mask)

            adjust(bk, points_to_adjust)

def visualize(bk, dataset_path):
    print("Starting visualization!")
    now = time.time()
    server = viser.ViserServer()

    points = np.asarray(bk.Tpts, dtype=float).reshape(-1, 3)
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

    K = np.array([
        [1660.076384971925, 0, 763.9663384634628],
        [0, 1656.285062933074, 986.3176281647676],
        [0, 0, 1],
    ])
    dist_coeffs = np.array([
        0.250022, -1.49939, -0.00296536, 0.00109332, 2.83634
    ])

    if use_stored:
        with open('geometry.pkl', 'rb') as file:
            bk = pickle.load(file)
    else:
        bk = Bookkeeper(K, dist_coeffs)
        sfm(bk, dataset_path, dataset_name)

        with open("geometry.pkl", "wb") as file:
            pickle.dump(bk, file)

    print(bk)

    # densify(bk)

    visualize(bk, dataset_path)

if __name__ == '__main__':
    main()
