import numpy as np
import cv2
import torch
from models.matching import Matching
from models.utils import frame2tensor
from opencv_detectors import Detector
from draw_matches import *


class SuperPoint(Detector):

    def __init__(self, use_magsac=False):
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2
            }
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.matching = Matching(config).eval().to(self.device)
        self.use_magsac = use_magsac

        # Camera intrinsics
        self.K = np.array([
            [456.558777441547, 0, 257.333104219938],
            [0, 452.348350048387, 256.926917124585],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([-0.0033, -0.2590, 0, 0, 0])


    def match_images(self, image1, image2):
        """
        Function to match two images using SuperPoint and SuperGlue.
        """

        keys = ['keypoints', 'scores', 'descriptors']

        # Load and process the first image
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image_tensor1 = frame2tensor(image1, self.device)
        data1 = self.matching.superpoint({'image': image_tensor1})
        data1 = {k + '0': data1[k] for k in keys}
        data1['image0'] = image_tensor1

        # Load and process the second image
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        image_tensor2 = frame2tensor(image2, self.device)
        data2 = self.matching.superpoint({'image': image_tensor2})
        data2 = {k + '1': data2[k] for k in keys}
        data2['image1'] = image_tensor2

        # Match the two images
        pred = self.matching({**data1, **data2})
        kpts0 = data1['keypoints0'][0].cpu().numpy()
        kpts1 = data2['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        scores = pred['matching_scores0'][0].cpu().detach().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        return kpts0, kpts1, mkpts0, mkpts1, scores[valid]


    def show_matches(self, frame1, frame2, undistort=True, vertical=False, idx=0, name=''):
        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and descriptors
        kp1, kp2, mkpts1, mkpts2, scores = self.match_images(frame1, frame2)


        draw_matches_sp(frame1, kp1, mkpts1, frame2, kp2, mkpts2, savename=f'matches\\{name}\\{idx}',
                     vertical=vertical)


    def match_pair(self, frame1, frame2, undistort=False, min_pts=10):
        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and matches
        kp1, kp2, mkpts1, mkpts2, scores = self.match_images(frame1, frame2)

        if len(kp1) < min_pts or len(kp2) < min_pts:
            return 'few_features', len(mkpts1)
        elif len(mkpts1) < min_pts:
            return 'few_matches', len(mkpts1)
        else:
            return 'success', len(mkpts1)


    def test_pair(self, frame1, frame2, gt_pose, undistort=True, min_pts=10):
        # method = 'SIFT'

        if undistort:
            # undistort frames
            frame1 = cv2.undistort(frame1, self.K, self.dist_coeffs, None, self.K)
            frame2 = cv2.undistort(frame2, self.K, self.dist_coeffs, None, self.K)

        # Find the keypoints and descriptors
        kp1, kp2, mkpts1, mkpts2, scores = self.match_images(frame1, frame2)

        if len(kp1) < min_pts or len(kp2) < min_pts:
            return 'few_features', None

        if len(mkpts1) < min_pts or len(mkpts2) < min_pts:
            return 'few_matches', None

        if self.use_magsac:
            return self.eval_magsac(mkpts1, mkpts2, gt_pose, scores)
        else:
            return self.eval_ransac(mkpts1, mkpts2, gt_pose)