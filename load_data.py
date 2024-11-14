import cv2
import pandas as pd
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def get_frame(root, id):

    img_path = os.path.join(root, f'{id}.jpg')
    img = cv2.imread(img_path)

    emt_path = os.path.join(root, f'gt.csv')
    emt = pd.read_csv(emt_path).iloc[id]

    # Get position
    pos = np.array([emt['X'], emt['Y'], emt['Z']]).T
    # Get rotation
    angles = np.array([emt['Roll'], emt['Pitch'], emt['Yaw']]).T
    rotation = R.from_euler('xyz', [angles[0], angles[1], angles[2]], degrees=True).as_matrix()

    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = pos

    return img, pose


def get_pair(root, id1, id2):

    img1, pose1 = get_frame(root, id1)
    img2, pose2 = get_frame(root, id2)
    relative_pose = np.linalg.inv(pose2) @ pose1

    return img1, img2, relative_pose