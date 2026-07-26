import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def auto_enhance_eye_image(image_pil_or_cv):
    """
    Ophthalmic Medical Image Auto-Improver:
    1. Converts image to LAB color space.
    2. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on L (Luminance) channel.
    3. Normalizes illumination glare and sharpens ocular lens opacities.
    """
    if isinstance(image_pil_or_cv, Image.Image):
        img_np = np.array(image_pil_or_cv)
    else:
        img_np = image_pil_or_cv

    # Ensure RGB format
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    # 1. Convert to LAB color space for adaptive contrast enhancement
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Re-merge LAB channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 3. Apply subtle edge sharpening for corneal and cataract cloudiness detail
    enhanced_pil = Image.fromarray(enhanced_np)
    enhanced_pil = enhanced_pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

    return enhanced_pil

if __name__ == "__main__":
    # Test auto enhancer on a sample image
    test_path = "sample_test_images/cataract/clinical_cataract_0056.jpg"
    if os.path.exists(test_path):
        img = Image.open(test_path)
        enhanced = auto_enhance_eye_image(img)
        enhanced.save("enhanced_sample_test.jpg")
        print("✅ Auto-Improver successfully processed test eye photo!")
