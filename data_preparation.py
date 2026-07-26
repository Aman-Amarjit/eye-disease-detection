import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter

def create_sample_dataset(base_dir="dataset", num_samples_per_class=100):
    """
    Creates sample dataset ONLY if no real clinical images are present in dataset/raw.
    """
    classes = ["normal", "cataract"]
    img_size = (224, 224)

    # Check if real images already exist
    has_real_normal = os.path.exists(os.path.join(base_dir, "raw", "normal")) and len(os.listdir(os.path.join(base_dir, "raw", "normal"))) > 0
    has_real_cataract = os.path.exists(os.path.join(base_dir, "raw", "cataract")) and len(os.listdir(os.path.join(base_dir, "raw", "cataract"))) > 0

    if has_real_normal and has_real_cataract:
        print("✅ Real clinical images detected in dataset/raw. Skipping synthetic sample generation.")
        return

    print("📁 Creating dataset directories...")
    for cls in classes:
        os.makedirs(os.path.join(base_dir, "raw", cls), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "processed", "train", cls), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "processed", "test", cls), exist_ok=True)
    
    print(f"👁️ Generating {num_samples_per_class} sample images per class for initial pipeline test...")
    
    for cls in classes:
        raw_cls_dir = os.path.join(base_dir, "raw", cls)
        for i in range(num_samples_per_class):
            img_path = os.path.join(raw_cls_dir, f"{cls}_{i+1:04d}.jpg")
            if not os.path.exists(img_path):
                # Generate eye mock canvas (Sclera + Iris + Pupil / Lens)
                img = Image.new('RGB', img_size, color=(240, 235, 230))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                
                # Outer eye / sclera ellipse
                draw.ellipse([20, 40, 204, 184], fill=(245, 245, 250), outline=(80, 50, 40), width=3)
                
                # Iris (brown / blue / hazel random variant)
                iris_color = (random.randint(40, 90), random.randint(30, 70), random.randint(20, 60))
                draw.ellipse([67, 67, 157, 157], fill=iris_color)
                
                if cls == "normal":
                    # Clear dark pupil
                    draw.ellipse([87, 87, 137, 137], fill=(10, 10, 15))
                    # Light reflection shine dot
                    draw.ellipse([110, 95, 120, 105], fill=(255, 255, 255))
                else:
                    # Cataract pupil (cloudy / opaque milky white-grey lens opacity)
                    draw.ellipse([87, 87, 137, 137], fill=(180, 190, 200))
                    # Cloudiness gradient / texture lines
                    for _ in range(15):
                        rx = random.randint(90, 130)
                        ry = random.randint(90, 130)
                        draw.ellipse([rx, ry, rx+8, ry+8], fill=(220, 225, 235, 150))
                    draw.ellipse([110, 95, 120, 105], fill=(255, 255, 255))
                
                img.save(img_path, quality=90)

    print("✅ Sample raw dataset generated successfully.")

def split_and_preprocess_dataset(base_dir="dataset", train_ratio=0.8, img_size=(224, 224)):
    """
    Splits dataset into train (80%) and test (20%), resizes images to 224x224,
    and organizes them into clean train/test folders.
    """
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    
    classes = ["normal", "cataract"]
    train_count = 0
    test_count = 0
    
    print("\n🔄 Processing & Splitting Dataset (80% Train / 20% Test)...")
    
    for cls in classes:
        cls_raw_path = os.path.join(raw_dir, cls)
        if not os.path.exists(cls_raw_path):
            continue
            
        images = [f for f in os.listdir(cls_raw_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        # Process Train
        for img_name in train_imgs:
            src = os.path.join(cls_raw_path, img_name)
            dst = os.path.join(processed_dir, "train", cls, img_name)
            _preprocess_single_image(src, dst, img_size)
            train_count += 1
            
        # Process Test
        for img_name in test_imgs:
            src = os.path.join(cls_raw_path, img_name)
            dst = os.path.join(processed_dir, "test", cls, img_name)
            _preprocess_single_image(src, dst, img_size)
            test_count += 1

    print(f"📊 Dataset Split Completed:")
    print(f"   • Training images: {train_count}")
    print(f"   • Testing images:  {test_count}")
    print(f"   • Target Resolution: {img_size[0]}x{img_size[1]}")

def _preprocess_single_image(src_path, dst_path, img_size=(224, 224)):
    """Resizes and saves image."""
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            img.save(dst_path, quality=92)
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    create_sample_dataset()
    split_and_preprocess_dataset()
