import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from Bookkeeper import Bookkeeper

import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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
        A[2 * i[mask], (view_indices[mask] - 1) * 6 + s] = 1
        A[2 * i[mask] + 1, (view_indices[mask] - 1) * 6 + s] = 1
        
    # Point params
    for s in range(3):
        col = (n_views - 1) * 6 + point_indices * 3 + s
        A[2 * i, col] = 1
        A[2 * i + 1, col] = 1
    return A

def bundle_adjustment(bk):
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

    f0 = compute_residuals(
        x0,
        n_views,
        n_points,
        view_indices,
        point_indices,
        points_2d,
        bk.K,
    )
    
    # 4. Solve
    n_views = len(view_ids)
    n_points = len(unique_pt_ids)
    A = get_sparsity_matrix(n_views, n_points, view_indices, point_indices)
    
    res = least_squares(
        compute_residuals, x0, jac_sparsity=A, x_scale='jac', ftol=1e-4, method='trf',
        args=(n_views, n_points, view_indices, point_indices, points_2d, bk.K)
    )

    print(f"Initial residual: {np.sqrt(np.mean(f0** 2, axis=0) )}")
    print(f"Final residual: {np.sqrt(np.mean(res.fun** 2, axis=0) )}")
    
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



# def bundle_adjustment(bk: Bookkeeper):


#     x0 = pack_params(camera_params, points_3d)
#     A = bundle_adjustment_sparsity(n_views, n_points, view_indices, point_indices)
#     f0 = compute_residuals(
#         x0,
#         n_views,
#         n_points,
#         view_indices,
#         point_indices,
#         points_2d,
#         bk.K,
#         fixed_camera_param,
#     )
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
#     res = least_squares(
#         compute_residuals,
#         x0,
#         jac_sparsity=A,
#         verbose=0,
#         x_scale="jac",
#         ftol=1e-6,  # FIXME:
#         args=(
#             n_views,
#             n_points,
#             view_indices,
#             point_indices,
#             points_2d,
#             bk.K,
#             fixed_camera_param,
#         ),
#     )

#     print(f"Initial residual: {np.sqrt(np.mean(f0** 2, axis=0) )}")
#     print(f"Final residual: {np.sqrt(np.mean(res.fun** 2, axis=0) )}")





# def bundle_adjustment_sparsity(
#     n_views, n_points, view_indices, point_indices, num_camera_params=6
# ):
#     view_indices = np.asarray(view_indices, dtype=int)
#     point_indices = np.asarray(point_indices, dtype=int)

#     n_observations = view_indices.size
#     n_rows = n_observations * 2

#     n_free_cameras = max(n_views - 1, 0)
#     n_camera_params = n_free_cameras * num_camera_params
#     n_point_params = n_points * 3
#     n_cols = n_camera_params + n_point_params

#     sparsity = lil_matrix((n_rows, n_cols), dtype=int)

#     obs_idx = np.arange(n_observations)

#     free_camera_mask = view_indices > 0
#     free_obs_idx = obs_idx[free_camera_mask]
#     free_camera_indices = view_indices[free_camera_mask] - 1

#     for param_offset in range(num_camera_params):
#         cols = free_camera_indices * num_camera_params + param_offset
#         sparsity[2 * free_obs_idx, cols] = 1
#         sparsity[2 * free_obs_idx + 1, cols] = 1

#     point_base = n_camera_params
#     for coord_offset in range(3):
#         cols = point_base + point_indices * 3 + coord_offset
#         sparsity[2 * obs_idx, cols] = 1
#         sparsity[2 * obs_idx + 1, cols] = 1

#     return sparsity


# def pack_params(camera_params, points_3d) -> np.ndarray:
#     params = np.hstack([camera_params[1:].ravel(), points_3d.ravel()])
#     return params


# def unpack_params(
#     params, n_views: int, n_points: int, fixed_camera_param: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray]:
#     n_free_cameras = max(n_views - 1, 0)
#     n_camera_params = n_free_cameras * 6
#     free_camera_params = params[:n_camera_params].reshape((n_free_cameras, 6))
#     camera_params = np.vstack([fixed_camera_param.reshape(1, 6), free_camera_params])
#     points_3d = params[n_camera_params:].reshape((n_points, 3))

#     return camera_params, points_3d


# def compute_residuals(
#     params,
#     n_views,
#     n_points,
#     view_indices,
#     point_indices,
#     points_2d,
#     K,
#     fixed_camera_param,
# ):
#     """Compute residuals.

#     `params` contains camera parameters and 3-D coordinates.
#     """
#     camera_params, points_3d = unpack_params(
#         params, n_views, n_points, fixed_camera_param
#     )
#     points_proj = project(points_3d[point_indices], camera_params[view_indices], K)
#     return (points_proj - points_2d).ravel()


# def rotate(points, rot_vecs):
#     """Rotate points by given rotation vectors.

#     Rodrigues' rotation formula is used.
#     """
#     theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
#     with np.errstate(invalid="ignore"):
#         v = rot_vecs / theta
#         v = np.nan_to_num(v)
#     dot = np.sum(points * v, axis=1)[:, np.newaxis]
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)

#     return (
#         cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
#     )


# def project(points, C, K):
#     """Convert 3-D points to 2-D by projecting onto images."""
#     points_proj = rotate(points, C[:, :3])
#     points_proj += C[:, 3:6]
#     points_hom = (K @ points_proj.T).T
    
#     homogeneous_depth = points_hom[:, 2]
#     safe_depth = np.where(
#         np.abs(homogeneous_depth) > 1e-12,
#         homogeneous_depth,
#         np.where(homogeneous_depth < 0, -1e-12, 1e-12),
#     )

#     return points_hom[:, :2] / safe_depth[:, np.newaxis]


# def get_observations(
#     bk: Bookkeeper, min_point_observations: int = 2
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     observed_cameras_by_point = {}
#     for point_index, view_index in bk.Tobs:
#         if 0 <= point_index < len(bk.Tpts):
#             observed_cameras_by_point.setdefault(point_index, set()).add(view_index)

#     optimized_point_indices = np.array(
#         [
#             point_index
#             for point_index, view_indices in sorted(observed_cameras_by_point.items())
#             if len(view_indices) >= min_point_observations
#         ],
#         dtype=int,
#     )
#     local_point_indices = {
#         point_index: local_index
#         for local_index, point_index in enumerate(optimized_point_indices)
#     }

#     view_indices = []
#     point_indices = []
#     points_2d = []
#     for (point_index, view_index), xy in bk.Tobs.items():
#         if point_index not in local_point_indices:
#             continue
#         view_indices.append(view_index)
#         point_indices.append(local_point_indices[point_index])
#         points_2d.append(np.asarray(xy, dtype=float).reshape(2))

#     return (
#         np.asarray(view_indices, dtype=int),
#         np.asarray(point_indices, dtype=int),
#         np.asarray(points_2d, dtype=float),
#         optimized_point_indices,
#     )


# def get_flattened_camera_params(bk: Bookkeeper) -> np.ndarray:
#     camera_params = []
#     for C in bk.Tvws.values():
#         R = C[:3, :3]; t = C[:, 3]
#         rvec, _ = cv.Rodrigues(R)  # [3, 3] -> [3, 1]
#         camera_param = np.hstack([rvec.flatten(), t.flatten()])
#         camera_params.append(camera_param)
#     return np.array(camera_params)


# def update_bookkeeping(
#     bk: Bookkeeper,
#     camera_params_optimized,
#     points_3d_optimized,
#     optimized_point_indices=None,
#     fixed_camera_indices=None,
# ):
#     fixed_camera_indices = (
#         set() if fixed_camera_indices is None else set(fixed_camera_indices)
#     )

#     if optimized_point_indices is None:
#         bk.Tpts = list(points_3d_optimized)
#     else:
#         points_3d = list(bk.Tpts)
#         for local_index, point_index in enumerate(optimized_point_indices):
#             points_3d[int(point_index)] = points_3d_optimized[local_index]
#         bk.points_3D = points_3d

#     for i, camera_param in enumerate(camera_params_optimized):
#         if i in fixed_camera_indices:
#             continue
#         rvec = camera_param[:3]
#         t = camera_param[3:6].reshape(3, 1)
#         R, _ = cv.Rodrigues(rvec)
#         bk.Tvws[i] = np.hstack((R, t))


# def bundle_adjustment(
#     bk: Bookkeeper,
#     num_points_to_adjust: int,
#     min_point_observations: int = 2,
# ):
#     """
#     Bundle adjustment.
#     """
#     # points_3d_all = np.asarray(bk.Tpts[:num_points_to_adjust], dtype=float)
#     points_3d_all = np.asarray(bk.Tpts, dtype=float)
#     n_views = len(bk.Tvws)
#     if n_views == 0:
#         raise ValueError("Bundle adjustment requires at least one view.")

#     view_indices, point_indices, points_2d, optimized_point_indices = get_observations(
#         bk, min_point_observations=min_point_observations
#     )
#     points_3d = points_3d_all[optimized_point_indices]
#     n_points = len(points_3d)

#     camera_params = get_flattened_camera_params(bk)
#     fixed_camera_param = camera_params[0].copy()

#     if n_points == 0 or len(view_indices) == 0:
#         print(
#             "Bundle adjustment skipped: no points have enough observations "
#             f"({min_point_observations=})."
#         )
#         return

#     x0 = pack_params(camera_params, points_3d)
#     A = bundle_adjustment_sparsity(n_views, n_points, view_indices, point_indices)
#     f0 = compute_residuals(
#         x0,
#         n_views,
#         n_points,
#         view_indices,
#         point_indices,
#         points_2d,
#         bk.K,
#         fixed_camera_param,
#     )
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
#     res = least_squares(
#         compute_residuals,
#         x0,
#         jac_sparsity=A,
#         verbose=0,
#         x_scale="jac",
#         ftol=1e-6,  # FIXME:
#         args=(
#             n_views,
#             n_points,
#             view_indices,
#             point_indices,
#             points_2d,
#             bk.K,
#             fixed_camera_param,
#         ),
#     )

#     print(f"Initial residual: {np.sqrt(np.mean(f0** 2, axis=0) )}")
#     print(f"Final residual: {np.sqrt(np.mean(res.fun** 2, axis=0) )}")

#     camera_params_optimized, points_3d_optimized = unpack_params(
#         res.x, n_views, n_points, fixed_camera_param
#     )
#     update_bookkeeping(
#         bk,
#         camera_params_optimized,
#         points_3d_optimized,
#         optimized_point_indices=optimized_point_indices,
#         fixed_camera_indices={0},
#     )
