import cv2
import numpy as np
from sklearn.cluster import KMeans
from grabcut_utils import remove_background_grabcut


# ---------- Leaf extraction ----------
def remove_background(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([15, 20, 20])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 10)

    result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return result, mask


# ---------- Disease segmentation ----------
def segment_disease_regions(img_bgr, leaf_mask, k):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    l, a, b = cv2.split(lab)
    leaf_pixels = leaf_mask > 0

    # Method 1: Darkness
    masked_gray = gray.copy()
    masked_gray[~leaf_pixels] = 255
    _, mask1 = cv2.threshold(
        masked_gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    mask1 &= leaf_mask

    # Method 2: KMeans
    features = np.column_stack([l[leaf_pixels], a[leaf_pixels], b[leaf_pixels]])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)

    label_img = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    label_img[leaf_pixels] = labels

    clusters = []
    for i in range(k):
        pix = (label_img == i) & leaf_pixels
        if np.sum(pix) > 0:
            clusters.append((i, np.mean(l[pix])))

    clusters.sort(key=lambda x: x[1])

    mask2 = np.zeros_like(gray)
    for i in range(min(2, len(clusters))):
        mask2[label_img == clusters[i][0]] = 255

    # Method 3: Color heuristic
    mask3 = np.zeros_like(gray)
    mask3[(a > 128) & (l < 100) & leaf_pixels] = 255

    # Combine
    vote = (
        (mask1 > 0).astype(int) +
        (mask2 > 0).astype(int) +
        (mask3 > 0).astype(int)
    )

    final_mask = np.zeros_like(gray)
    final_mask[vote >= 2] = 255

    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, 2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, 1)

    seg_img = cv2.bitwise_and(img_bgr, img_bgr, mask=final_mask)
    return final_mask, seg_img, label_img


def calculate_disease_percentage(disease_mask, leaf_mask):
    leaf_area = np.sum(leaf_mask > 0)
    disease_area = np.sum(disease_mask > 0)
    return (disease_area / leaf_area * 100) if leaf_area else 0


# ---------- Full pipeline ----------
def detect_disease(img_bgr, k):
    _, img = remove_background_grabcut(img_bgr)
    img = cv2.resize(img, (256, 256))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    img = cv2.cvtColor(
        cv2.merge([clahe.apply(l), a, b]),
        cv2.COLOR_Lab2BGR
    )

    _, leaf_mask = remove_background(img)
    disease_mask, seg_img, clusters = segment_disease_regions(img, leaf_mask, k)
    percentage = calculate_disease_percentage(disease_mask, leaf_mask)

    return img, leaf_mask, disease_mask, seg_img, clusters, percentage
