import cv2 as cv
import numpy as np
import time
import viser
import viser.transforms as vtf

from Bookkeeper import Bookkeeper

def visualize(bk: Bookkeeper, dataset_path: str):
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
