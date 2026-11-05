import cv2
import numpy as np


def detect_red_lesions(
    image_bgr: np.ndarray,
    retina_mask: np.ndarray = None,
    vessels_mask: np.ndarray = None,  # <-- Ajout du masque de vaisseaux
    min_area: int = 5,
    max_area: int = 1000
) -> np.ndarray:
    # 1. Extraction du canal vert et Top-Hat
    green = image_bgr[:, :, 1] if image_bgr.ndim == 3 else image_bgr
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Élimination des vaisseaux sanguins
    if vessels_mask is not None:
        # On ne garde que ce qui N'EST PAS un vaisseau
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(vessels_mask))

    # 3. Application du masque de la rétine (FOV)
    if retina_mask is not None:
        binary = cv2.bitwise_and(binary, retina_mask.astype(np.uint8))

    # 4. Filtrage par taille
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
            clean_mask[labels == i] = 255

    return clean_mask