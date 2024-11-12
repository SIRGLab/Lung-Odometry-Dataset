import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('TkAgg')


def draw_matches(img1, keypoints1, img2, keypoints2, matches, savename=None, vertical=False):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10) if vertical else (10, 5))

    # Determine the size and layout based on 'vertical' flag
    if vertical:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        view[:h1, :w1, :] = img1
        view[h1:h1+h2, :w2, :] = img2
    else:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        view[:h1, :w1, :] = img1
        view[:h2, w1:w1+w2, :] = img2

    if not vertical:
        view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

    # Show the image
    ax.imshow(view)

    # Sets to keep track of matched keypoints
    matched_keypoints1 = set()
    matched_keypoints2 = set()

    # Draw the matches
    for m in matches:
        pt1 = keypoints1[m.queryIdx].pt
        pt2 = keypoints2[m.trainIdx].pt

        if vertical:
            pt2 = (int(pt2[0]), int(pt2[1] + h1))  # Adjust y-coordinate for the second image if vertical
        else:
            pt2 = (int(pt2[0] + w1), int(pt2[1]))  # Adjust x-coordinate for the second image if horizontal

        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), lw=0.5)
        ax.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), marker='o', s=2)

        # Add to the matched set
        matched_keypoints1.add(m.queryIdx)
        matched_keypoints2.add(m.trainIdx)

    # Scatter unmatched keypoints
    for i, kp in enumerate(keypoints1):
        if i not in matched_keypoints1:# and not is_in_top_left_corner(kp.pt):
            ax.scatter(kp.pt[0], kp.pt[1], c=(1, 0, 0), marker='o', s=2)

    for i, kp in enumerate(keypoints2):
        if i not in matched_keypoints2:# and not is_in_top_left_corner(kp.pt):
            if vertical:
                ax.scatter(kp.pt[0], kp.pt[1] + h1, c=(1, 0, 0), marker='o', s=2)
            else:
                ax.scatter(kp.pt[0] + w1, kp.pt[1], c=(1, 0, 0), marker='o', s=2)

    # Remove axes and save
    ax.axis('off')
    if savename is not None:
        plt.savefig(f'{savename}.png', bbox_inches='tight')
    plt.close(fig)
    plt.show()


def draw_matches_loftr(img1, kp1, img2, kp2, savename=None, vertical=False):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10) if vertical else (10, 5))

    # Determine the size and layout based on 'vertical' flag
    if vertical:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        view[:h1, :w1, :] = img1
        view[h1:h1+h2, :w2, :] = img2
    else:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        view[:h1, :w1, :] = img1
        view[:h2, w1:w1+w2, :] = img2

    if not vertical:
        view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

    # Show the image
    ax.imshow(view)

    # Draw the matches
    for pt1, pt2 in zip(kp1, kp2):
        if vertical:
            pt2 = (pt2[0], pt2[1] + h1)  # Adjust y-coordinate for the second image if vertical
        else:
            pt2 = (pt2[0] + w1, pt2[1])  # Adjust x-coordinate for the second image if horizontal

        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), lw=0.2)
        ax.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), s=1, marker='o')

    # Remove axes and save
    ax.axis('off')
    if savename is not None:
       plt.savefig(f'{savename}.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def draw_matches_sp(img1, kp1, mkp1, img2, kp2, mkp2, savename=None, vertical=False):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10) if vertical else (10, 5))

    # Determine the size and layout based on 'vertical' flag
    if vertical:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        view[:h1, :w1, :] = img1
        view[h1:h1+h2, :w2, :] = img2
    else:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        view[:h1, :w1, :] = img1
        view[:h2, w1:w1+w2, :] = img2

    if not vertical:
        view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)

    # Draw the matches
    for pt1, pt2 in zip(mkp1, mkp2):
        if vertical:
            pt2 = (pt2[0], pt2[1] + h1)  # Adjust y-coordinate for the second image if vertical
        else:
            pt2 = (pt2[0] + w1, pt2[1])  # Adjust x-coordinate for the second image if horizontal

        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), lw=0.2)
        ax.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), s=1, marker='o')

    # Draw the remaining kps
    for pt1, pt2 in zip(kp1, kp2):
        if vertical:
            pt2 = (pt2[0], pt2[1] + h1)  # Adjust y-coordinate for the second image if vertical
        else:
            pt2 = (pt2[0] + w1, pt2[1])  # Adjust x-coordinate for the second image if horizontal

        ax.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), s=1, marker='o')
    for pt1, pt2 in zip(mkp1, mkp2):
        if vertical:
            pt2 = (pt2[0], pt2[1] + h1)  # Adjust y-coordinate for the second image if vertical
        else:
            pt2 = (pt2[0] + w1, pt2[1])  # Adjust x-coordinate for the second image if horizontal

        ax.scatter([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=(0, 1, 0), s=1, marker='o')

    # Show the image
    ax.imshow(view)
    ax.axis('off')
    plt.show()
    plt.close(fig)
