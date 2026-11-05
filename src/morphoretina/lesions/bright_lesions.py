import cv2
import numpy as np


def detect_bright_lesions(
    image_bgr: np.ndarray,
    retina_mask: np.ndarray = None,
    optic_disc_mask: np.ndarray = None,
    min_area: int = 10,
    max_area: int = 5000
) -> np.ndarray:
    """Détecte les lésions claires en éliminant la papille / disque optique."""
    green = image_bgr[:, :, 1] if image_bgr.ndim == 3 else image_bgr

    # 1. CLAHE & White Top-Hat (structures brillantes)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # 2. Binarisation
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Soustraction du disque optique si disponible
    if optic_disc_mask is not None:
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(optic_disc_mask.astype(np.uint8)))

    # 4. Masque Rétine
    if retina_mask is not None:
        kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded_mask = cv2.erode(retina_mask.astype(np.uint8), kernel_mask)
        binary = cv2.bitwise_and(binary, binary, mask=eroded_mask)

    # 5. Filtrage par surface
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean_mask = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean_mask[labels == i] = 255

    return clean_mask