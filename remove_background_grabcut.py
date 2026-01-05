import cv2
import numpy as np

def grabcut(img_bgr):
    mask = np.zeros(img_bgr.shape[:2], np.uint8)

    h, w = img_bgr.shape[:2]
    rect = (10, 10, w-20, h-20)  # assume object is roughly centered

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Foreground mask
    mask2 = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype('uint8')

    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask2)

    return mask2, result