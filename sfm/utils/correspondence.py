import cv2 as cv
import numpy as np

def get_best_points(kp, desc, n_percent):
    combined = sorted(zip(kp, desc), key=lambda x: x[0].response, reverse=True)
    top_n = combined[:int(len(combined) * n_percent / 100)]

    new_kp = [x[0] for x in top_n]
    new_desc = np.asarray([x[1] for x in top_n], dtype=np.float32)
    return new_kp, new_desc

def interest_points(img, mask, n_percent=100):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)

    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, mask)

    kp, desc = get_best_points(kp, desc, n_percent)

    return kp, desc

def match_points(kp1, kp2, desc1, desc2, n_percent=100):
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    safe = sorted([m for m, k in matches if m.distance < 0.75 * k.distance], key=lambda x: x.distance)
    safe = safe[:int(len(safe) * n_percent / 100)]

    idx1 = [m.queryIdx for m in safe]
    idx2 = [m.trainIdx for m in safe]

    sel_kp1 = [kp1[i] for i in idx1]
    sel_kp2 = [kp2[i] for i in idx2]
    sel_desc1 = desc1[idx1]
    sel_desc2 = desc2[idx2]

    return sel_kp1, sel_kp2, sel_desc1, sel_desc2, idx1, idx2

def find_2d_correspondences(img1, img2, mask1, mask2, p_points, p_matches):
    kp1, desc1 = interest_points(img1, mask1, p_points)
    kp2, desc2 = interest_points(img2, mask2, p_points)

    return match_points(kp1, kp2, desc1, desc2, p_matches)

def find_2d_3d_correspondences(indices_3d, descs_3d, desc2, ratio_threshold=0.8):
    if len(desc2) == 0 or len(descs_3d) < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    matcher = cv.BFMatcher(cv.NORM_L2)
    k = min(32, len(descs_3d))
    matches = matcher.knnMatch(desc2, descs_3d, k=k)

    best_match_by_point = {}
    for query_matches in matches:
        if len(query_matches) < 2:
            continue

        best_match = query_matches[0]
        best_point_index = int(indices_3d[best_match.trainIdx])
        second_best_other_point = None

        for candidate in query_matches[1:]:
            candidate_point_index = int(indices_3d[candidate.trainIdx])
            if candidate_point_index != best_point_index:
                second_best_other_point = candidate
                break

        if (
            second_best_other_point is not None
            and best_match.distance >= ratio_threshold * second_best_other_point.distance
        ):
            continue

        current_best = best_match_by_point.get(best_point_index)
        if current_best is None or best_match.distance < current_best.distance:
            best_match_by_point[best_point_index] = best_match

    if not best_match_by_point:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    point_indices = np.array(sorted(best_match_by_point.keys()), dtype=int)
    query_indices = np.array(
        [best_match_by_point[point_index].queryIdx for point_index in point_indices],
        dtype=int,
    )

    return point_indices, query_indices
