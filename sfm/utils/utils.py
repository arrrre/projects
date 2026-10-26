import cv2 as cv
import numpy as np

def fmatrix_from_cameras(C1, C2):
    """Fundamental matrix from camera pair
    
    Parameters
    ------------------
    C1 : (3, 4) array
        Camera 1
    C2 : (3, 4) array
        Camera 2
        
    Returns
    ---------------------
    F : (3,3) array
        Fundamental matrix corresponding to C1 and C2
    """
    U, s, V = np.linalg.svd(C2) # Note: C2 = U S V  (V already transposed)
    n = V[3, :]
    e = np.dot(C1, n)
    C2pinv = np.linalg.pinv(C2)
    F = np.dot(cross_matrix(e), np.dot(C1, C2pinv))
    return F

def cross_matrix(v):
    v = v.ravel()
    if not v.size == 3:
        raise ValueError("Can only handle 3D vectors")

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def triangulate(C1, C2, K1, K2, dist_coeffs1, dist_coeffs2, x1, x2):
    x1_und = undistort(x1, K1, dist_coeffs1)
    x2_und = undistort(x2, K2, dist_coeffs2)

    P1 = K1 @ C1
    P2 = K2 @ C2

    F = fmatrix_from_cameras(P1, P2)

    pts1 = x1_und.reshape(1, -1, 2).astype(np.float32)
    pts2 = x2_und.reshape(1, -1, 2).astype(np.float32)

    pts1, pts2 = cv.correctMatches(F, pts1, pts2)

    X_hom = cv.triangulatePoints(P1, P2, pts1.reshape(-1, 2).T, pts2.reshape(-1, 2).T)

    return X_hom[:3] / X_hom[3]

def undistort(points, K, dist_coeffs):
    points_cv = points.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv.undistortPoints(points_cv, K, dist_coeffs, P=K)
    return undistorted.reshape(-1, 2)

def robust_pnp(
    corr2d,
    corr3d,
    K,
    dist_coeffs,
    N=10000,
    threshold=5.0,
):
    object_points = corr3d.astype(np.float32)
    image_points = corr2d.astype(np.float32)

    success, rvec, tvec, inliers = cv.solvePnPRansac(
        object_points,
        image_points,
        K,
        dist_coeffs,
        iterationsCount=N,
        reprojectionError=threshold,
        confidence=0.999,
        flags=cv.SOLVEPNP_EPNP,
    )

    if not success or inliers is None or len(inliers) < 4:
        print(f"Sample 2D point: {corr2d[0, :]}")
        print(f"Sample 3D point: {corr3d[0, :]}")
        raise ValueError("PnP RANSAC failed to find at least four inliers.")

    inlier_indices = inliers.ravel()
    success, rvec, tvec = cv.solvePnP(
        object_points[inlier_indices],
        image_points[inlier_indices],
        K,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise ValueError("PnP refinement failed.")

    R, _ = cv.Rodrigues(rvec)

    return R, tvec, inlier_indices
