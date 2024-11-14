import numpy as np
import cv2
import torch
from kornia.feature import LoFTR
from opencv_detectors import Detector
from draw_matches import *


class LoFTR_Detector(Detector):

    def __init__(self):
        self.loftr = LoFTR(pretrained='outdoor')
        self.loftr = self.loftr.cuda() if torch.cuda.is_available() else self.loftr

        # Camera intrinsics
        self.K = np.array([
            [456.558777441547, 0, 257.333104219938],
            [0, 452.348350048387, 256.926917124585],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([-0.0033, -0.2590, 0, 0, 0])


    def match_images(self, frame1, frame2):
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        t1 = torch.from_numpy(frame1_gray)[None][None] / 255.
        t2 = torch.from_numpy(frame2_gray)[None][None] / 255.

        input = {"image0": t1, "image1": t2}
        res = self.loftr(input)

        return res['keypoints0'].cpu().numpy(), res['keypoints1'].cpu().numpy()


    def show_matches(self, frame1, frame2, undistort=True):
        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and matches
        kp1, kp2 = self.match_images(frame1, frame2)

        draw_matches_loftr(frame1, kp1, frame2, kp2)


    def match_pair(self, frame1, frame2, undistort=False, min_pts=10):
        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and matches
        kp1, kp2 = self.match_images(frame1, frame2)

        if len(kp1) < min_pts or len(kp2) < min_pts:
            return 'few_matches', len(kp1)
        else:
            return 'success', len(kp1)


    def test_pair(self, frame1, frame2, gt_pose, undistort=True, min_pts=10):

        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and descriptors
        kp1, kp2 = self.match_images(frame1, frame2)

        if len(kp1) < min_pts or len(kp2) < min_pts:
            return 'few_matches', None

        return self.eval_ransac(kp1, kp2, gt_pose)