from dataclasses import dataclass
import numpy as np


@dataclass
class LesionDetectionResult:
    """Résultat de la détection de lésions sur une image rétinienne."""
    mask_red_lesions: np.ndarray       
    mask_bright_lesions: np.ndarray    
    count_red_lesions: int
    count_bright_lesions: int
    area_red_lesions_px: int
    area_bright_lesions_px: int