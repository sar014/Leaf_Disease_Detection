import cv2
import numpy as np
from sklearn.cluster import KMeans
from grabcut_utils import remove_background_grabcut


# ---------- Disease segmentation ----------
def segment_disease_regions(img_bgr, leaf_mask, k):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    l, a, b = cv2.split(lab)
    leaf_pixels = leaf_mask > 0

    # Method 1: Darkness - Otsu
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
        if np.any(pix) > 0:
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
    img_bgr = cv2.resize(img_bgr, (256, 256))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Combines the three LAB channels back into one image, an image in LAB format with improved contrast
    enhanced_img = cv2.cvtColor(
        cv2.merge([l_enhanced, a, b]),
        cv2.COLOR_Lab2BGR
    )

    # ---------- Step 3: GrabCut background removal ----------
    leaf_mask, leaf_img = remove_background_grabcut(enhanced_img)

    # Ensure binary mask
    leaf_mask = (leaf_mask > 0).astype(np.uint8) * 255

    # ---------- Step 4: Disease segmentation ----------
    disease_mask, seg_img, cluster_labels = segment_disease_regions(
        leaf_img, leaf_mask, k
    )

    # ---------- Step 5: Disease percentage ----------
    percentage = calculate_disease_percentage(disease_mask, leaf_mask)

    return (
        enhanced_img,
        leaf_mask,
        disease_mask,
        seg_img, # seg_img → visualized disease-only image
        cluster_labels, #label_img → K-Means cluster label map
        percentage
    )
