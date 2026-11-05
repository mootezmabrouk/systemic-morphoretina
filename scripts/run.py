import sys
from pathlib import Path

# Ajout du dossier src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from morphoretina.anatomy.retina_mask import normalize_scale, retina_mask
from morphoretina.anatomy.optic_disc import find_optic_disc
from morphoretina.anatomy.vessels import segment_vessels
from morphoretina.preprocessing.preprocess import preprocess
from morphoretina.lesions.pipeline import extract_all_lesions


def clean_optic_disc_mask(disc_mask: np.ndarray) -> np.ndarray:
    """Conserve uniquement la composante connexe la plus volumineuse (la vraie papille)."""
    if disc_mask is None:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(disc_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    lab, n = ndi.label(opened)
    if n == 0:
        return disc_mask
    sizes = ndi.sum(opened, lab, range(1, n + 1))
    best_label = 1 + int(np.argmax(sizes))
    return lab == best_label


def main():
    parser = argparse.ArgumentParser(description="Pipeline complet MorphoRetina")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image")
    args = parser.parse_args()

    # 1. Chargement et échelle
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Image introuvable : {args.image}")

    small_bgr, _ = normalize_scale(image_bgr, 1420.0)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    # 2. Prétraitement
    cfg_prep = {
        "preprocessing": {
            "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8]},
            "denoise": {"n_orientations": 4, "line_length": 5}
        }
    }
    denoised_green = preprocess(small_rgb, cfg_prep)["denoised"]

    # 3. Extraction anatomique
    fov_mask = retina_mask(small_rgb)
    disc_res = find_optic_disc(denoised_green, fov_mask)
    disc_mask = clean_optic_disc_mask(disc_res["mask"]) if disc_res else None
    vessels_mask = segment_vessels(denoised_green, fov_mask, mode="range")["vessels"]

    # 4. Pipeline global de détection des lésions
    resultat = extract_all_lesions(
        image_bgr=small_bgr,
        retina_mask=fov_mask,
        vessels_mask=vessels_mask,
        optic_disc_mask=disc_mask
    )

    # 5. Affichage console
    print(f"\n--- RÉSULTATS POUR : {args.image} ---")
    print(f"Lésions rouges  : {resultat.count_red_lesions} objets ({resultat.area_red_lesions_px} px)")
    print(f" Lésions claires : {resultat.count_bright_lesions} objets ({resultat.area_bright_lesions_px} px)")

    # 6. Affichage graphique Matplotlib (Grille 2x3)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(small_rgb)
    axes[0, 0].set_title("1. Image Échelonnée")

    axes[0, 1].imshow(fov_mask, cmap="gray")
    axes[0, 1].set_title("2. Masque FOV")

    axes[0, 2].imshow(disc_mask if disc_mask is not None else fov_mask * 0, cmap="gray")
    axes[0, 2].set_title("3. Disque Optique (Nettoyé)")

    axes[1, 0].imshow(vessels_mask, cmap="gray")
    axes[1, 0].set_title("4. Vaisseaux Sanguins")

    axes[1, 1].imshow(resultat.mask_red_lesions, cmap="Reds")
    axes[1, 1].set_title(f"5. Lésions Rouges ({resultat.count_red_lesions})")

    axes[1, 2].imshow(resultat.mask_bright_lesions, cmap="YlOrBr")
    axes[1, 2].set_title(f"6. Lésions Claires ({resultat.count_bright_lesions})")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()