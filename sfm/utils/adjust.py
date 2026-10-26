import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from Bookkeeper import Bookkeeper

def rotate(points, rot_vecs):
    """Rotate points by rotation vectors (Rodrigues)."""
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    v = np.divide(rot_vecs, theta, out=np.zeros_like(rot_vecs), where=theta!=0)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params, K):
    """Projects 3D points into 2D pixel coordinates."""
    # camera_params: [rvec_x, rvec_y, rvec_z, tx, ty, tz]
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    
    # Standard pinhole projection
    points_hom = (K @ points_proj.T).T
    return points_hom[:, :2] / points_hom[:, 2, np.newaxis]

def compute_residuals(params, n_views, n_points, view_indices, point_indices, points_2d, K):
    """Main objective function for least_squares."""
    # Unpack
    n_cam_params = (n_views - 1) * 6
    # First camera is fixed at [0,0,0, 0,0,0] (identity/origin)
    cameras = np.zeros((n_views, 6))
    cameras[1:] = params[:n_cam_params].reshape((n_views - 1, 6))
    points_3d = params[n_cam_params:].reshape((n_points, 3))
    
    # Project and calculate error
    points_proj = project(points_3d[point_indices], cameras[view_indices], K)
    return (points_proj - points_2d).ravel()

def get_sparsity_matrix(n_views, n_points, view_indices, point_indices):
    """Speeds up BA by telling the solver which params affect which residuals."""
    m = view_indices.size * 2
    n = (n_views - 1) * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(view_indices.size)
    # Camera params (only for views > 0)
    for s in range(6):
        mask = view_indices > 0
        col = (view_indices[mask] - 1) * 6 + s
        A[2 * i[mask], col] = 1
        A[2 * i[mask] + 1, col] = 1

    # Point params
    for s in range(3):
        col = (n_views - 1) * 6 + point_indices * 3 + s
        A[2 * i, col] = 1
        A[2 * i + 1, col] = 1
    return A

def bundle_adjustment(bk: Bookkeeper):
    # 1. Map frame_idx and point_idx to local 0...N range
    view_ids = sorted(bk.Tvws.keys())
    view_map = {v_id: i for i, v_id in enumerate(view_ids)}
    
    # We only optimize points that have observations
    obs_keys = list(bk.Tobs.keys()) # (frame_idx, point_idx)
    unique_pt_ids = sorted(list(set(k[1] for k in obs_keys)))
    pt_map = {pt_id: i for i, pt_id in enumerate(unique_pt_ids)}
    
    # 2. Gather observations
    view_indices = np.array([view_map[k[0]] for k in obs_keys])
    point_indices = np.array([pt_map[k[1]] for k in obs_keys])
    points_2d = np.array([bk.Tobs[k] for k in obs_keys])
    
    # 3. Initial parameters
    # Extract [rvec, t] for all cameras except the first
    cam_params = []
    for v_id in view_ids[1:]:
        pose = bk.Tvws[v_id] # [R | t]
        rvec, _ = cv.Rodrigues(pose[:3, :3])
        cam_params.append(np.hstack([rvec.flatten(), pose[:, 3]]))
    
    pts_3d = np.array([bk.Tpts[p_id] for p_id in unique_pt_ids])
    
    x0 = np.hstack([np.array(cam_params).ravel(), pts_3d.ravel()])
    
    # 4. Solve
    n_views = len(view_ids)
    n_points = len(unique_pt_ids)
    A = get_sparsity_matrix(n_views, n_points, view_indices, point_indices)
    
    res = least_squares(
        compute_residuals, x0, jac_sparsity=A, x_scale='jac', ftol=1e-4, method='trf',
        args=(n_views, n_points, view_indices, point_indices, points_2d, bk.K)
    )

    f0 = compute_residuals(
        x0,
        n_views,
        n_points,
        view_indices,
        point_indices,
        points_2d,
        bk.K,
    )

    print(f"Old residual: {np.sqrt(np.mean(f0**2, axis=0))}")
    print(f"New residual: {np.sqrt(np.mean(res.fun**2, axis=0))}")
    
    # 5. Push results back to Bookkeeper
    n_cam_params = (n_views - 1) * 6
    new_cams = res.x[:n_cam_params].reshape((-1, 6))
    new_pts = res.x[n_cam_params:].reshape((-1, 3))
    
    for i, v_id in enumerate(view_ids[1:]):
        rvec = new_cams[i, :3]
        t = new_cams[i, 3:].reshape(3, 1)
        R, _ = cv.Rodrigues(rvec)
        bk.Tvws[v_id] = np.hstack([R, t])
        
    for i, pt_id in enumerate(unique_pt_ids):
        bk.Tpts[pt_id] = new_pts[i]

    return res
