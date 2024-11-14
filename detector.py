import numpy as np
import cv2
#import pymagsac
from eval_pose import *


class Detector():

    def __init__(self):
        self.K = np.array([
            [456.558777441547, 0, 257.333104219938],
            [0, 452.348350048387, 256.926917124585],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([-0.0033, -0.2590, 0, 0, 0])

    def test_pair(self, frame1, frame2, gt_pose):
        return NotImplementedError

    def eval_ransac(self, kp1, kp2, gt_pose):

        E, mask = cv2.findEssentialMat(kp1, kp2, self.K, method=cv2.RANSAC, prob=0.9999, threshold=0.5)

        if mask.sum() < 5:
            return 'ransac_fail', None

        inliers = np.array(mask).reshape(-1)
        inlier_pts1 = kp1[inliers == 1]
        inlier_pts2 = kp2[inliers == 1]

        # Get angle errors
        a_e, b_e = self.essential_errors(E, inlier_pts1, inlier_pts2, gt_pose)
        # Check homography
        a_hom, b_hom = self.homography_errors(inlier_pts1, inlier_pts2, gt_pose)

        return 'success', [a_e, b_e, a_hom, b_hom]

    def eval_magsac(self, kp1, kp2, gt_pose, scores):

        return self.eval_ransac(kp1, kp2, gt_pose)
        # correspondences = np.float32([(list(kp1[i]) + list(kp2[i])) for i in range(len(kp1))]).reshape(-1, 4)
        #
        # E, mask = pymagsac.findEssentialMatrix(
        #     np.ascontiguousarray(correspondences),
        #     np.ascontiguousarray(self.K),
        #     np.ascontiguousarray(self.K),
        #     480, 480, 480, 480,
        #     probabilities=scores,
        #     sampler=4,
        #     use_magsac_plus_plus=True,
        #     sigma_th=1.25)
        #
        # if mask.sum() < 5:
        #     return 'ransac_fail', None
        #
        # inliers = np.array(mask).reshape(-1)
        # inlier_pts1 = kp1[inliers == 1]
        # inlier_pts2 = kp2[inliers == 1]
        #
        # # Get angle errors
        # a_e, b_e = self.essential_errors(E, inlier_pts1, inlier_pts2, gt_pose)
        # # Check homography
        # a_hom, b_hom = self.homography_errors(inlier_pts1, inlier_pts2, gt_pose)
        #
        # return 'success', [a_e, b_e, a_hom, b_hom]

    def essential_errors(self, E, inlier_pts1, inlier_pts2, gt_pose):

        R_true = gt_pose[:3, :3]
        t_true = gt_pose[:3, 3]

        _, R, t, mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, self.K)

        a_e = evaluate_rotation(R, R_true)
        b_e = angle_between(t.flatten(), t_true)

        return a_e, b_e

    def homography_errors(self, inlier_pts1, inlier_pts2, gt_pose):

        R_true = gt_pose[:3, :3]
        t_true = gt_pose[:3, 3]

        H, mask = cv2.findHomography(inlier_pts1, inlier_pts2, method=0)
        if mask.sum() < 5:
            return np.nan, np.nan

        retval, Rs, Ts, normals = cv2.decomposeHomographyMat(H, self.K)

        a_hom = 180
        b_hom = 180

        for hom_idx in range(len(Rs)):
            R = Rs[hom_idx]
            t = Ts[hom_idx]
            a = evaluate_rotation(R, R_true)
            b = angle_between(t.flatten(), t_true)

            if max(a, b) < max(a_hom, b_hom):
                a_hom = a
                b_hom = b

        return a_hom, b_hom