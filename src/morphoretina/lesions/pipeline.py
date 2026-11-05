import cv2
import numpy as np
from morphoretina.lesions.base import LesionDetectionResult
from morphoretina.lesions.bright_lesions import detect_bright_lesions
from morphoretina.lesions.red_lesion import detect_red_lesions


def extract_all_lesions(
    image_bgr: np.ndarray,
    retina_mask: np.ndarray = None,
    vessels_mask: np.ndarray = None,
    optic_disc_mask: np.ndarray = None
) -> LesionDetectionResult:
    """Pipeline globale de détection de lésions."""
    # 1. Détection des 2 types de lésions
    mask_red = detect_red_lesions(
        image_bgr, 
        retina_mask=retina_mask, 
        vessels_mask=vessels_mask
    )
    mask_bright = detect_bright_lesions(
        image_bgr, 
        retina_mask=retina_mask, 
        optic_disc_mask=optic_disc_mask
    )

    # 2. Calcul des comptages
    num_red, _, _, _ = cv2.connectedComponentsWithStats(mask_red)
    num_bright, _, _, _ = cv2.connectedComponentsWithStats(mask_bright)

    count_red = max(0, num_red - 1)
    count_bright = max(0, num_bright - 1)

    # 3. Calcul des surfaces
    area_red = int(np.sum(mask_red > 0))
    area_bright = int(np.sum(mask_bright > 0))

    return LesionDetectionResult(
        mask_red_lesions=mask_red,
        mask_bright_lesions=mask_bright,
        count_red_lesions=count_red,
        count_bright_lesions=count_bright,
        area_red_lesions_px=area_red,
        area_bright_lesions_px=area_bright,
    )