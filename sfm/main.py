import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
import pickle

from Bookkeeper import Bookkeeper

from utils.adjust import bundle_adjustment
from utils.correspondence import interest_points, match_points, find_2d_correspondences, find_2d_3d_correspondences
from utils.utils import triangulate, robust_pnp
from utils.visualize import visualize

def init(bk, frame_idx1, frame_idx2, img1, img2, mask1, mask2):
    """
    Initialization of incremental SfM according to algorithm 21.1 in IREG.
    """
    kp1, kp2, desc1, desc2 = find_2d_correspondences(img1, img2, mask1, mask2, 90, 90)
    corr1 = np.asarray([kp.pt for kp in kp1])
    corr2 = np.asarray([kp.pt for kp in kp2])

    E, mask = cv.findEssentialMat(corr1, corr2, cameraMatrix=bk.K, method=cv.RANSAC)
    _, R, t, mask = cv.recoverPose(E, corr1, corr2, cameraMatrix=bk.K, mask=mask)
    mask = mask.ravel().astype(bool)

    corr1 = corr1[mask]; corr2 = corr2[mask]
    desc1 = desc1[mask]; desc2 = desc2[mask]

    C1 = np.eye(3, 4)
    C2 = np.hstack((R, t))

    bk.add_view(frame_idx1, C1);
    bk.add_view(frame_idx2, C2)

    points_3d = triangulate(C1, C2, bk.K, bk.K, bk.dist_coeffs, bk.dist_coeffs, corr1, corr2)
    for i, p_3d in enumerate(points_3d.T):
        p_idx = bk.add_point(p_3d)
        bk.add_obs(frame_idx1, p_idx, corr1[i], desc1[i])
        bk.add_obs(frame_idx2, p_idx, corr2[i], desc2[i])

def add_new_view(bk, frame_idx1, frame_idx2, img1, img2, mask1, mask2):
    """
    Add new view to the SfM problem according to algorithm 21.2 in IREG.
    """
    kp1, desc1 = interest_points(img1, mask1, n_percent=90)
    kp2, desc2 = interest_points(img2, mask2, n_percent=90)

    indices_3d, descs_3d = bk.get_dcs()
    indices_3d, indices_2d = find_2d_3d_correspondences(indices_3d, descs_3d, desc2)

    corr2d = np.array([kp2[i].pt for i in indices_2d])
    desc2d = desc2[indices_2d]
    corr3d = np.array([bk.Tpts[i] for i in indices_3d])

    R2, t2, inlier_indices = robust_pnp(corr2d, corr3d, bk.K, bk.dist_coeffs)
    bk.add_view(frame_idx2, np.hstack((R2, t2)))

    for i in inlier_indices:
        bk.add_obs(frame_idx2, indices_3d[i], corr2d[i], desc2d[i])

    # New potential 3D points
    kp1m, kp2m, desc1m, desc2m, idx1, idx2 = match_points(kp1, kp2, desc1, desc2, n_percent=90)

    used_indices = set(indices_2d[inlier_indices])

    mask_new = [i not in used_indices for i in idx2]
    corr1 = np.array([p.pt for p in np.array(kp1m)[mask_new]])
    corr2 = np.array([p.pt for p in np.array(kp2m)[mask_new]])
    desc1 = np.array(desc1m)[mask_new]
    desc2 = np.array(desc2m)[mask_new]

    if len(corr1) <= 0:
        return

    C1 = bk.get_view(frame_idx1)
    C2 = bk.get_view(frame_idx2)
    R1 = C1[:3, :3]; t1 = C1[:, 3]
    R2 = C2[:3, :3]; t2 = C2[:, 3]

    rvec1, _ = cv.Rodrigues(R1)
    rvec2, _ = cv.Rodrigues(R2)

    X_new = triangulate(C1, C2, bk.K, bk.K, bk.dist_coeffs, bk.dist_coeffs, corr1, corr2)

    proj1, _ = cv.projectPoints(X_new, rvec1, t1, bk.K, bk.dist_coeffs)
    proj2, _ = cv.projectPoints(X_new, rvec2, t2, bk.K, bk.dist_coeffs)

    proj1 = proj1.reshape(-1, 2)
    proj2 = proj2.reshape(-1, 2)

    errs1 = np.linalg.norm(proj1 - corr1, axis=1)
    errs2 = np.linalg.norm(proj2 - corr2, axis=1)

    valid_indices = np.where((errs1 < 5.0) & (errs2 < 5.0))[0]

    for i in valid_indices:
        p_3d = X_new[:, i]
        p_idx = bk.add_point(p_3d)
        bk.add_obs(frame_idx1, p_idx, corr1[i], desc1[i])
        bk.add_obs(frame_idx2, p_idx, corr2[i], desc2[i])

def adjust(bk):
    """
    Select inliers: T'points
    Bundle adjustment
    Re-triangulate
    Remove outliers
    """
    bundle_adjustment(bk)

def sfm(bk: Bookkeeper, dataset_path: Path, dataset_name: str, num_images: int = 36):
    mask = cv.imread(dataset_path / f"{dataset_name}_mask.png")

    for i in range(num_images - 1):
        frame_idx1 = i + 1
        frame_idx2 = i + 2
        img1 = cv.imread(dataset_path / f"frame{frame_idx1:02d}.png")
        img2 = cv.imread(dataset_path / f"frame{frame_idx2:02d}.png")

        if i == 0:
            init(bk, frame_idx1, frame_idx2, img1, img2, mask, mask)
        else:
            add_new_view(bk, frame_idx1, frame_idx2, img1, img2, mask, mask)

            adjust(bk)

        print(f"{i=}, {bk=}")

def main():
    parser = argparse.ArgumentParser(description="Structure from Motion (SfM)")
    parser.add_argument("--dataset_name", type=str, default="my")
    parser.add_argument("--dataset_path", type=str, default="C:/Arvid/code/tsbb33-datasets/")
    parser.add_argument("--use_stored", action="store_true", default=False)
    parser.add_argument("--pickle_filename", type=str, default="bk.pkl")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = Path(args.dataset_path) / args.dataset_name
    use_stored = args.use_stored
    pickle_filename = args.pickle_filename

    K = np.array([
        [1660.076384971925, 0, 763.9663384634628],
        [0, 1656.285062933074, 986.3176281647676],
        [0, 0, 1],
    ])
    dist_coeffs = np.array([
        0.250022, -1.49939, -0.00296536, 0.00109332, 2.83634
    ])

    if use_stored:
        with open(pickle_filename, 'rb') as file:
            bk = pickle.load(file)
    else:
        bk = Bookkeeper(K, dist_coeffs)
        sfm(bk, dataset_path, dataset_name)

        with open(pickle_filename, 'wb') as file:
            pickle.dump(bk, file)

    # densify(bk)

    visualize(bk, dataset_path)

if __name__ == '__main__':
    main()
