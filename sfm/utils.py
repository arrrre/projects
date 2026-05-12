import numpy as np
import cv2 as cv

def fmatrix_stls(pl, pr):
    """Estimate fundamental matrix using 8-point algorithm
    
    Parameters
    ---------------
    pl : (2, N) ndarray
        Left image coordinates
    pr : (2, N) ndarray
        Right image coordinates
        
    Returns
    --------------
    Fundamental matrix F
    """
    if not pl.shape == pr.shape:
        raise ValueError('pl and pr must have same shape')
    
    _, N = pl.shape
    
    def scaling_homography(x):
        xm = np.mean(x, axis=1)
        x_norm = x - xm.reshape(-1,1) # Remove mean
        L = np.sqrt(1. / 2. / N * np.sum(x_norm**2))
        H = np.array([[1. / L, 0        , -xm[0] / L],
                      [0.    , 1. / L   , -xm[1] / L],
                      [0.    , 0.       , 1.         ]])
        return H
    
    S = scaling_homography(pl)
    T = scaling_homography(pr)
    # Map points through S and T respectively
    def map_homography(p, H):    
        x = p[0]
        y = p[1]
        xh = x * H[0,0] + y * H[0,1] + H[0, 2]
        yh = x * H[1,0] + y * H[1,1] + H[1, 2]
        return xh, yh
    
    lhx, lhy = map_homography(pl, S)
    rhx, rhy = map_homography(pr, T)
    
    # Generate matrix A for which A * vec(F) = 0
    X, Y = lhx, lhy
    x, y = rhx, rhy
    A = np.vstack((X*x, X*y, X, Y*x, Y*y, Y, x, y, np.ones((1, N))))
    A = A.T
    
    U, s, V = np.linalg.svd(A); # Note: V already transposed
    Fs = V[-1, :].reshape(3,3)
    
    # Enforce rank 2
    U, s, V = np.linalg.svd(Fs) # Note: V already transposed
    D = np.diag(s)
    D[2,2] = 0
    Fs = np.dot(U, np.dot(D, V))
    
    # Undo scaling
    F = np.dot(S.T, np.dot(Fs, T))
    return F

def fmatrix_residuals(F, x, y):
    """Calculate residuals for fundamental matrix and image points
    
    Calculates the distances between the epipolar lines and the supplied points.
    
    Parameters
    -------------------
    F : (3,3) ndarray
        Fundamental matrix such that x.T * F * y = 0
    x : (2, N) ndarray
        Points in left image
    y : (2, N) ndarray
        Points in right image
    
    Returns
    -------------------
    residuals : (2, N) ndarray
        Signed distance from x (dim 0) and y (dim 1) to epipolar lines
    """
    if not x.shape == y.shape:
        raise ValueError('x and y must have same sizes')
    
    x = homog(x)
    y = homog(y)

    l1 = np.dot(F, y)
    l2 = np.dot(F.T, x)
    
    #normalize = lambda l: l / np.sqrt(l[0,:]**2 + l[1,:]**2)
    #l1 = normalize(l1)
    #l2 = normalize(l2) 
    l1s = np.sqrt(l1[0,:]**2 + l1[1,:]**2)
    l2s = np.sqrt(l2[0,:]**2 + l2[1,:]**2)
    
    #res1 = np.sum(l1 * x, axis=0)
    #res2 = np.sum(l2 * y, axis=0)
    res1 = np.sum(l1 * x, axis=0) / l1s
    res2 = np.sum(l2 * y, axis=0) / l2s
    
    return np.vstack((res1, res2))

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

def homog(x):
    if x.ndim == 1:
        return np.append(x, 1).reshape(-1, 1)

    return np.vstack([x, np.ones((1, x.shape[1]))])

def cross_matrix(v):
    v = v.ravel()
    if not v.size == 3:
        raise ValueError("Can only handle 3D vectors")

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def E_from_F(F, K1, K2):
    return K1.T @ F @ K2
    # return K2.T @ F @ K1

def R_t_from_E(E, K1, K2, x1, x2):
    U, S, V_T = np.linalg.svd(E)
    V = V_T.T

    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    C1 = np.eye(3, 4)

    R1 = V @ W.T @ U.T
    if np.linalg.det(R1) < 0:
        print(f"det(R1) < 0")
        R1 = V @ (-W.T) @ U.T

    R2 = V @ W @ U.T
    if np.linalg.det(R2) < 0:
        print(f"det(R2) < 0")
        R2 = V @ (-W) @ U.T

    t = V[:, 2].reshape(3, 1)

    potential_C2 = [
        np.hstack((R1, t)),
        np.hstack((R1, -t)),
        np.hstack((R2, t)),
        np.hstack((R2, -t)),
    ]

    for C2 in potential_C2:
        X1 = triangulate(C1, C2, K1, K2, x1, x2)
        R, t = np.hsplit(C2, [3])
        X2 = R @ X1 + t

        if (X1[2] > 0) and (X2[2] > 0):
            return R, t

def triangulate(C1, C2, K1, K2, x1, x2):
    x1_norm = np.linalg.inv(K1) @ homog(x1)
    x2_norm = np.linalg.inv(K2) @ homog(x2)

    pts1 = x1_norm[:2, :].T.reshape(1, -1, 2).astype(np.float32)
    pts2 = x2_norm[:2, :].T.reshape(1, -1, 2).astype(np.float32)

    F = fmatrix_from_cameras(C1, C2) 

    pts1_corr, pts2_corr = cv.correctMatches(F, pts1, pts2)

    X_hom = cv.triangulatePoints(C1, C2, pts1_corr.reshape(-1, 2).T, pts2_corr.reshape(-1, 2).T)

    return X_hom[:3] / X_hom[3]

def get_best_points(kp, des, n_percent):
    combined = sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)
    top_n = combined[:int(len(combined) * n_percent / 100)]

    new_kp = [x[0] for x in top_n]
    new_des = np.array([x[1] for x in top_n])
    return new_kp, new_des

def get_best_matches(kp1, kp2, des1, des2, n_percent):
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    safe = sorted([m for m, k in matches if m.distance < 0.75 * k.distance], key=lambda x: x.distance)
    safe = safe[:int(len(safe) * n_percent / 100)]

    pts1 = np.array([kp1[m.queryIdx].pt for m in safe]).T
    pts2 = np.array([kp2[m.trainIdx].pt for m in safe]).T

    sel_kp1 = [kp1[m.queryIdx] for m in safe]
    sel_kp2 = [kp2[m.trainIdx] for m in safe]
    sel_des1 = des1[[m.queryIdx for m in safe]]
    sel_des2 = des2[[m.trainIdx for m in safe]]

    return pts1, pts2, sel_kp1, sel_kp2, sel_des1, sel_des2

def find_correspondences(img1, img2, mask1, mask2, p_points, p_matches):
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
    _, mask1 = cv.threshold(mask1, 0, 255, cv.THRESH_BINARY)
    mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)
    _, mask2 = cv.threshold(mask2, 0, 255, cv.THRESH_BINARY)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, mask1)
    kp2, des2 = sift.detectAndCompute(img2, mask2)

    kp1, des1 = get_best_points(kp1, des1, p_points)
    kp2, des2 = get_best_points(kp2, des2, p_points)

    corr1, corr2, kp1, kp2, des1, des2 = get_best_matches(kp1, kp2, des1, des2, p_matches)

    return corr1, corr2, kp1, kp2, des1, des2

def robust_f(corr1, corr2, N=10000, threshold=1.25):
    curr_inliers = 0
    curr_indices = []
    curr_F = None

    for _ in range(N):
        random_indices = np.random.choice(corr1.shape[1], 8, replace=False)

        F_est = fmatrix_stls(corr1[:, random_indices], corr2[:, random_indices])

        residuals = fmatrix_residuals(F_est, corr1, corr2)
        distances = np.linalg.norm(residuals, axis=0)

        inliers_mask = distances < threshold
        num_inliers = np.sum(inliers_mask)

        if num_inliers > curr_inliers:
            curr_inliers = num_inliers
            curr_indices = inliers_mask
            curr_F = F_est

    return curr_F, curr_indices

def robust_pnp(corr1, corr2, corr3d, K, dist_coeffs, N=10000, threshold=2.25):
    curr_inliers = 0
    curr_indices = []
    curr_rvec = None
    curr_tvec = None

    for _ in range(N):
        random_indices = np.random.choice(corr1.shape[1], 3, replace=False)

        success, rvecs, tvecs = cv.solveP3P(
            corr3d[:, random_indices].T,
            corr2[:, random_indices].T,
            K,
            dist_coeffs,
            flags=cv.SOLVEPNP_P3P
        )

        for j in range(success):
            points_proj, _ = cv.projectPoints(corr3d.T, rvecs[j], tvecs[j], K, dist_coeffs)
            points_proj = points_proj.reshape(-1, 2)

            error = np.linalg.norm(points_proj - corr2.T, axis=1)
            inliers_mask = error < threshold
            num_inliers = np.sum(inliers_mask)

            if num_inliers > curr_inliers:
                curr_inliers = num_inliers
                curr_indices = inliers_mask
                curr_rvec = rvecs[j]
                curr_tvec = tvecs[j]

    R, _ = cv.Rodrigues(curr_rvec)
    t = curr_tvec

    if curr_inliers >= 4:
        success, rvec, tvec = cv.solvePnP(
            corr3d[:,curr_indices].T,
            corr2[:,curr_indices].T,
            K,
            dist_coeffs,
            rvec=curr_rvec,
            tvec=curr_tvec,
            useExtrinsicGuess=True,
            flags=cv.SOLVEPNP_ITERATIVE
        )
        R, _ = cv.Rodrigues(rvec)
        t = tvec

    return R, t, corr1[:, curr_indices], corr2[:, curr_indices], corr3d[:, curr_indices]

def robust_pnp(
    corr2,
    corr3d,
    point_indices,
    point_descriptors,
    K,
    dist_coeffs,
    N=10000,
    threshold=5.0,
):
    object_points = corr3d.T.astype(np.float32)
    image_points = corr2.T.astype(np.float32)

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
    return (
        R,
        tvec,
        corr2[:, inlier_indices],
        corr3d[:, inlier_indices],
        point_indices[inlier_indices],
        point_descriptors[inlier_indices],
    )

def bundle_adjustment(bk):
    pass
