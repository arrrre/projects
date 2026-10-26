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

    kp1m = [kp1[i] for i in idx1]
    kp2m = [kp2[i] for i in idx2]
    desc1m = desc1[idx1]
    desc2m = desc2[idx2]

    return kp1m, kp2m, desc1m, desc2m, idx1, idx2

def find_2d_correspondences(img1, img2, mask1, mask2, p_points, p_matches):
    kp1, desc1 = interest_points(img1, mask1, p_points)
    kp2, desc2 = interest_points(img2, mask2, p_points)

    kp1m, kp2m, desc1m, desc2m, _, _ = match_points(kp1, kp2, desc1, desc2, p_matches)

    return kp1m, kp2m, desc1m, desc2m

def find_2d_3d_correspondences(indices_3d, descs_3d, desc2, ratio_threshold=0.8):
    matcher = cv.BFMatcher(cv.NORM_L2)
    matches = matcher.knnMatch(desc2, descs_3d, k=min(32, len(descs_3d)))

    selected_3d_indices = []
    selected_2d_indices = []

    for m_list in matches:
        if len(m_list) < 2: continue

        best_m = m_list[0]
        best_pt_idx = indices_3d[best_m.trainIdx]

        second_best_m = None
        for next_m in m_list[1:]:
            if indices_3d[next_m.trainIdx] != best_pt_idx:
                second_best_m = next_m
                break

        if second_best_m is None: continue

        if best_m.distance < ratio_threshold * second_best_m.distance:
            selected_3d_indices.append(best_pt_idx)
            selected_2d_indices.append(best_m.queryIdx)

    return np.array(selected_3d_indices), np.array(selected_2d_indices)
