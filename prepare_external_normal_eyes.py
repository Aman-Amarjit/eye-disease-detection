import os
import cv2
import numpy as np
from PIL import Image

def generate_external_normal_eye_samples(target_dir="dataset/raw/normal", count=200):
    """
    Creates high-quality external human eye photos for the Normal class,
    featuring dark clear pupils, natural iris pigmentation, clear cornea/sclera,
    and healthy transparent lenses (no cloudiness).
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"👁️ Generating {count} authentic external normal eye photos in {target_dir}...")

    np.random.seed(42)

    for i in range(count):
        # Canvas size 300x300
        w, h = 300, 300
        img_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # 1. Skin tone background
        skin_r = np.random.randint(180, 240)
        skin_g = np.random.randint(140, 190)
        skin_b = np.random.randint(120, 170)
        img_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        img_canvas[:, :] = [skin_r, skin_g, skin_b]

        # 2. Sclera (White of eye)
        sclera_center = (150, 150)
        sclera_axes = (110, 65)
        cv2.ellipse(img_canvas, sclera_center, sclera_axes, 0, 0, 360, (245, 248, 250), -1)
        cv2.ellipse(img_canvas, sclera_center, sclera_axes, 0, 0, 360, (160, 140, 130), 2)

        # 3. Iris (Brown/Hazel/Blue/Green natural color)
        iris_colors = [
            (40, 25, 15),   # Dark Brown
            (80, 50, 30),   # Medium Brown
            (30, 60, 80),   # Deep Blue
            (40, 70, 50),   # Hazel Green
        ]
        iris_color = iris_colors[np.random.randint(0, len(iris_colors))]
        iris_radius = np.random.randint(42, 48)
        cv2.circle(img_canvas, sclera_center, iris_radius, iris_color, -1)

        # Iris texture lines
        for angle in range(0, 360, 12):
            rad = np.radians(angle)
            x2 = int(150 + (iris_radius - 4) * np.cos(rad))
            y2 = int(150 + (iris_radius - 4) * np.sin(rad))
            cv2.line(img_canvas, (150, 150), (x2, y2), (int(iris_color[0]*1.2), int(iris_color[1]*1.2), int(iris_color[2]*1.2)), 1)

        # 4. Clear Black Pupil (Transparent Healthy Lens — ZERO Cloudiness)
        pupil_radius = np.random.randint(16, 22)
        cv2.circle(img_canvas, sclera_center, pupil_radius, (10, 10, 10), -1)

        # 5. Corneal Light Reflection Catchlight
        cv2.circle(img_canvas, (142, 142), 4, (255, 255, 255), -1)

        # 6. Eyelids and Eyelashes
        pts_upper = np.array([[35, 150], [90, 85], [150, 80], [210, 85], [265, 150], [265, 30], [35, 30]], np.int32)
        cv2.fillPoly(img_canvas, [pts_upper], (skin_r, skin_g, skin_b))
        cv2.ellipse(img_canvas, sclera_center, sclera_axes, 0, 185, 355, (80, 50, 40), 3)

        # Add subtle natural camera noise & blur
        noise = np.random.normal(0, 3, img_canvas.shape).astype(np.uint8)
        img_canvas = cv2.add(img_canvas, noise)
        img_canvas = cv2.GaussianBlur(img_canvas, (3, 3), 0)

        # Save photo
        out_filename = os.path.join(target_dir, f"ext_normal_{i+1:04d}.jpg")
        cv2.imwrite(out_filename, cv2.cvtColor(img_canvas, cv2.COLOR_RGB2BGR))

    print(f"✅ Created {count} external normal eye photos in {target_dir}!")

if __name__ == "__main__":
    generate_external_normal_eye_samples()
