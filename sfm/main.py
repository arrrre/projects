import argparse
import cv2 as cv
import numpy as np

K = np.array([
    [1660.076384971925, 0, 763.9663384634628],
    [0, 1656.285062933074, 986.3176281647676],
    [0, 0, 1],
])

dist_coeffs = np.array([0.250022, -1.49939, -0.00296536, 0.00109332, 2.83634])

NUM_IMAGES = 36

class Bookkeeper:
    def __init__(self):
        self.Tvws = {} # views
        self.Tpts = [] # 3D points
        self.Tobs = {} # observations

        # Usage: view_idx = add_view()
        """
        Usage:
        bk = Bookkeeper()
        view_idx = bk.add_view(frame_idx, pose)
        point_idx = bk.add_point(coord3D)
        bk.add_obs(coord2D, view_idx, point_idx)        
        """

    def __repr__(self):
        return f"Bookkeeper(Views: {len(self.Tvws)}, Points: {len(self.Tpts)}, Obs: {len(self.Tobs)})"

    def add_view(self, frame_idx, pose):
        self.Tvws[frame_idx] = pose
        return frame_idx
    
    def add_point(self, coord):
        self.Tpts.append(coord)
        return len(self.Tpts) - 1
    
    def add_obs(self, coord, view_idx, point_idx):
        self.Tobs[(view_idx, point_idx)] = coord

def init(bk, img1, img2, mask, frame_idx1, frame_idx2):
    bk.add_view(frame_idx1, np.eye(3, 4))
    bk.add_view(frame_idx2, np.random.rand(3, 4))

def loop(bk, img1, img2, mask, frame_idx1, frame_idx2):
    bk.add_view(frame_idx2, np.random.rand(3, 4))

def main():
    parser = argparse.ArgumentParser(description="SfM")
    parser.add_argument("--dataset_name", type=str, default="my")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = f"C:/Arvid/code/tsbb33-datasets/{dataset_name}/"

    mask = cv.imread(f"{dataset_path}{dataset_name}_mask.png")

    bk = Bookkeeper()

    for i in range(NUM_IMAGES):
        print(f"{i=}")
        frame_idx1 = i + 1
        frame_idx2 = i + 2 if i < NUM_IMAGES - 1 else 1
        img1 = cv.imread(f"{dataset_path}frame{frame_idx1:02d}.png")
        img2 = cv.imread(f"{dataset_path}frame{frame_idx2:02d}.png")

        if i == 0:
            init(bk, img1, img2, mask, frame_idx1, frame_idx2)
        else:
            loop(bk, img1, img2, mask, frame_idx1, frame_idx2)

    print(bk)
    print(bk.Tvws)

if __name__ == '__main__':
    main()
