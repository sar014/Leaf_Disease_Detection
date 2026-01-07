import cv2
import numpy as np

def remove_background_grabcut(img_bgr):
    """
    Removes background using GrabCut.
    Returns foreground mask and foreground image.
    """
    mask = np.zeros(img_bgr.shape[:2], np.uint8)

    h, w = img_bgr.shape[:2]
    rect = (10, 10, w - 20, h - 20)

    # Internal models used by GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img_bgr,
        mask,
        rect,
        bgdModel,
        fgdModel,
        5, # Number of refinement iterations
        cv2.GC_INIT_WITH_RECT
    )
    
    # a clean binary foreground mask
    mask2 = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype("uint8")

    # GrabCut mask values
    # GC_BGD → definite background
    # GC_PR_BGD → probable background
    # GC_FGD → definite foreground
    # GC_PR_FGD → probable foreground

    # Extract the foreground image
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask2)

    return mask2, result
