import numpy as np
import cv2
from detector import Detector
from eval_pose import *
from draw_matches import *

class CVDetectors(Detector):

    def __init__(self, method='sift'):
        self.method = method
        self.extractor = None

        if method == 'sift':
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        elif method == 'orb':
            self.detector = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        elif method == 'akaze':
            self.detector = cv2.AKAZE_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        elif method == 'brisk':
            self.detector = cv2.BRISK_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == 'brisk-freak':
            self.detector = cv2.BRISK_create()
            self.extractor = cv2.xfeatures2d.FREAK_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError('Invalid detector name.')

        # Camera intrinsics
        self.K = np.array([
            [456.558777441547, 0, 257.333104219938],
            [0, 452.348350048387, 256.926917124585],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([-0.0033, -0.2590, 0, 0, 0])


    def detect_img(self, frame):
        if self.extractor is None:
            return self.detector.detectAndCompute(frame, None)
        else:
            keypoints = self.detector.detect(frame, None)
            return self.extractor.compute(frame, keypoints)


    def match_pair(self, frame1, frame2, undistort=False, min_pts=10):
        # method = 'SIFT'

        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and descriptors
        keypoints1, descriptors1 = self.detect_img(frame1)
        keypoints2, descriptors2 = self.detect_img(frame2)

        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return 'few_features', 0

        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)

        if len(keypoints1) < min_pts or len(keypoints2) < min_pts:
            return 'few_features', len(matches)

        if len(matches) == 0:
            return 'few_matches', 0

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        if len(good_matches) < min_pts:
            return 'few_matches', len(good_matches)
        else:
            return 'success', len(good_matches)


    def show_matches(self, frame1, frame2, undistort=True):
        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and descriptors
        keypoints1, descriptors1 = self.detect_img(frame1)
        keypoints2, descriptors2 = self.detect_img(frame2)

        matches = []
        if len(keypoints1) >= 2 and (len(keypoints2) >= 2):
            matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        else:
            good_matches = []

        draw_matches(frame1, keypoints1, frame2, keypoints2, good_matches)


    def test_pair(self, frame1, frame2, gt_pose, undistort=True, min_pts=10):
        # method = 'SIFT'

        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and descriptors
        keypoints1, descriptors1 = self.detect_img(frame1)
        keypoints2, descriptors2 = self.detect_img(frame2)

        if len(keypoints1) < min_pts or len(keypoints2) < min_pts:
            return 'few_features', None

        matches = self.bf.knnMatch(descriptors1, descriptors2, k=2)

        if len(matches) < min_pts:
            return 'few_matches', None

        # Apply ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        if len(good_matches) < min_pts:
            return 'few_matches', None

        kp1 = np.float32([keypoints1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 2)
        kp2 = np.float32([keypoints2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return self.eval_ransac(kp1, kp2, gt_pose)