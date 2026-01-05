import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from scipy import ndimage
from remove_background_grabcut import grabcut


# Remove everything except the leaf.
def remove_background(img_bgr):

    #Convert Image to HSV Color Space
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define range for green/yellow leaf colors
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])

    # Creates binary mask
    raw_mask = cv2.inRange(hsv, lower_green, upper_green)

    # keep only the largest connected component → main leaf
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    leaf_mask = np.zeros_like(raw_mask)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask[labels == largest_idx] = 255

    # Morphological operations to clean up mask
    kernel = np.ones((5,5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel, iterations=2) # dilation followed by erosion, Fills small holes inside the leaf

    # Apply mask
    result = cv2.bitwise_and(img_bgr, img_bgr, mask=leaf_mask)
    return result, leaf_mask

def segment_disease_regions(img_bgr, leaf_mask,k):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    leaf_pixels = leaf_mask > 0
    features = np.column_stack([
        L[leaf_pixels],
        A[leaf_pixels],
        B[leaf_pixels],
    ]).astype(np.float32)

    n_clusters = k
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)

    label_img = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    label_img[leaf_pixels] = labels

    # collect per-cluster stats
    cluster_stats = []
    for i in range(n_clusters):
        mask_i = (label_img == i) & leaf_pixels
        if not np.any(mask_i):
            continue
        mean_L = np.mean(L[mask_i])
        mean_A = np.mean(A[mask_i])
        mean_B = np.mean(B[mask_i])
        mean_S = np.mean(S[mask_i])
        cluster_stats.append(dict(
            cid=i, L=mean_L, A=mean_A, B=mean_B, S=mean_S
        ))

    # heuristic thresholds; you can tune these on a few images
    disease_clusters = []
    for cs in cluster_stats:
        # dark-ish
        is_dark = cs["L"] < 140
        # reddish / brownish (A > ~135 is more red than healthy green)
        is_reddish = cs["A"] > 135
        # reasonably saturated (avoid pure gray shadows)
        is_saturated = cs["S"] > 40

        if is_dark and is_reddish and is_saturated:
            disease_clusters.append(cs["cid"])

    # fallback: if nothing passes the rule, take the darkest reddish cluster
    if not disease_clusters:
        # sort by L, then prefer higher A
        cluster_stats.sort(key=lambda cs: (cs["L"], -cs["A"]))
        disease_clusters = [cluster_stats[0]["cid"]]

    disease_mask = np.zeros_like(label_img, dtype=np.uint8)
    for cid in disease_clusters:
        disease_mask[label_img == cid] = 255

    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)

    kernel = np.ones((3,3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    seg_img = cv2.bitwise_and(img_bgr, img_bgr, mask=disease_mask)
    return disease_mask, seg_img, label_img

def calculate_disease_percentage(disease_mask, leaf_mask):
    """Calculate percentage of leaf area affected by disease"""
    leaf_area = np.sum(leaf_mask > 0)
    disease_area = np.sum(disease_mask > 0)

    if leaf_area > 0:
        percentage = (disease_area / leaf_area) * 100
    else:
        percentage = 0

    return percentage, disease_area, leaf_area

def detect_disease(image_path, k):
    """Main detection pipeline with improved visualization"""
    img = cv2.imread(image_path)
    _,img = grabcut(img)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    img = cv2.resize(img, (256, 256))

    # Step 1: Contrast Enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)

    # Step 2: Remove Background
    img_no_bg, leaf_mask = remove_background(img_enhanced)

    # Step 3: Segment Disease Regions
    disease_mask, seg_img, cluster_img = segment_disease_regions(img_enhanced, leaf_mask,k)

    # Step 4: Calculate Disease Percentage
    percentage, disease_area, leaf_area = calculate_disease_percentage(disease_mask, leaf_mask)

    return {
        "percentage": percentage,
        "original": img,
        "enhanced": img_enhanced,
        "leaf_mask": leaf_mask,
        "cluster_img": cluster_img,
        "disease_mask": disease_mask,
        "segmented": seg_img
    }

