import os
import shutil
import zipfile
import urllib.request
import kagglehub

def download_real_cataract_dataset():
    print("=" * 80)
    print("📥 DOWNLOADING REAL CLINICAL CATARACT EYE DATASET (Kaggle Hub)")
    print("=" * 80)

    target_raw_dir = "dataset/raw"
    os.makedirs(os.path.join(target_raw_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(target_raw_dir, "cataract"), exist_ok=True)

    try:
        # Download real cataract dataset from KaggleHub
        # Dataset: jr2ngb/cataractdataset or kushal1506/cataract-dataset
        print("🔍 Requesting dataset 'jr2ngb/cataractdataset' from KaggleHub...")
        path = kagglehub.dataset_download("jr2ngb/cataractdataset")
        print(f"✅ Real dataset downloaded to local cache: {path}")

        # Scan for normal and cataract directories in downloaded dataset
        normal_src = None
        cataract_src = None

        for root, dirs, files in os.walk(path):
            for d in dirs:
                d_lower = d.lower()
                if "normal" in d_lower:
                    normal_src = os.path.join(root, d)
                elif "cataract" in d_lower:
                    cataract_src = os.path.join(root, d)

        # Copy real clinical images into workspace dataset structure
        if normal_src and cataract_src:
            print(f"📂 Copying real Normal eye images from: {normal_src}")
            norm_imgs = [f for f in os.listdir(normal_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in norm_imgs:
                shutil.copy(os.path.join(normal_src, img), os.path.join(target_raw_dir, "normal", img))

            print(f"📂 Copying real Cataract eye images from: {cataract_src}")
            cat_imgs = [f for f in os.listdir(cataract_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in cat_imgs:
                shutil.copy(os.path.join(cataract_src, img), os.path.join(target_raw_dir, "cataract", img))

            print(f"\n🎉 REAL CLINICAL DATASET INTEGRATED SUCCESSFULLY:")
            print(f"   • Real Normal Eye Images:   {len(norm_imgs)}")
            print(f"   • Real Cataract Eye Images: {len(cat_imgs)}")
            return True

    except Exception as e:
        print(f"⚠️ KaggleHub download note: {e}")

    # Fallback to public GitHub medical eye dataset mirror if Kaggle authentication is needed
    print("\n🌐 Fetching alternative real clinical eye dataset from public GitHub repository mirror...")
    mirror_url = "https://github.com/dineshdharme/Cataract-Detection/archive/refs/heads/master.zip"
    zip_path = "real_dataset.zip"

    try:
        urllib.request.urlretrieve(mirror_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("temp_dataset")

        # Organize extracted real clinical images
        for root, dirs, files in os.walk("temp_dataset"):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, f)
                    if "cataract" in root.lower() or "cataract" in f.lower():
                        shutil.copy(file_path, os.path.join(target_raw_dir, "cataract", f))
                    elif "normal" in root.lower() or "normal" in f.lower():
                        shutil.copy(file_path, os.path.join(target_raw_dir, "normal", f))

        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists("temp_dataset"):
            shutil.rmtree("temp_dataset")

        norm_count = len(os.listdir(os.path.join(target_raw_dir, "normal")))
        cat_count = len(os.listdir(os.path.join(target_raw_dir, "cataract")))

        print(f"✅ REAL CLINICAL EYE DATASET LOADED FROM PUBLIC MIRROR:")
        print(f"   • Real Normal Images:   {norm_count}")
        print(f"   • Real Cataract Images: {cat_count}")
        return True

    except Exception as e:
        print(f"❌ Error fetching real dataset: {e}")
        return False

if __name__ == "__main__":
    download_real_cataract_dataset()
