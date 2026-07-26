import os
import shutil
import zipfile
import urllib.request

def fetch_real_clinical_dataset():
    print("=" * 80)
    print("📥 DOWNLOADING AUTHENTIC CLINICAL CATARACT DATASET FROM GITHUB")
    print("=" * 80)

    target_raw_dir = "dataset/raw"
    os.makedirs(os.path.join(target_raw_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(target_raw_dir, "cataract"), exist_ok=True)

    # Real Medical Dataset mirrors on GitHub
    mirrors = [
        "https://github.com/SohhamSeal/Cloudy-Eyes/archive/refs/heads/main.zip",
        "https://github.com/emeryntumba/cataract-classification/archive/refs/heads/master.zip",
        "https://github.com/rainyNighti/CSDI/archive/refs/heads/main.zip"
    ]

    for url in mirrors:
        print(f"\n🌐 Attempting download from: {url}")
        zip_path = "real_dataset_temp.zip"
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                
            print("📦 Extracting clinical image files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("temp_real_dataset")

            cat_count = 0
            norm_count = 0

            # Scan and extract real medical eye photographs
            for root, dirs, files in os.walk("temp_real_dataset"):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, f)
                        r_lower = root.lower()
                        f_lower = f.lower()
                        
                        if any(k in r_lower or k in f_lower for k in ["cataract", "cloudy", "disease", "opacity"]):
                            cat_count += 1
                            dst = os.path.join(target_raw_dir, "cataract", f"clinical_cataract_{cat_count:04d}.jpg")
                            shutil.copy(file_path, dst)
                        elif any(k in r_lower or k in f_lower for k in ["normal", "clear", "healthy", "non_cataract"]):
                            norm_count += 1
                            dst = os.path.join(target_raw_dir, "normal", f"clinical_normal_{norm_count:04d}.jpg")
                            shutil.copy(file_path, dst)

            # Clean temporary files
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists("temp_real_dataset"):
                shutil.rmtree("temp_real_dataset")

            if cat_count > 0 or norm_count > 0:
                print(f"\n🎉 REAL CLINICAL EYE DATASET INTEGRATED SUCCESSFULLY:")
                print(f"   • Real Clinical Cataract Images: {cat_count}")
                print(f"   • Real Clinical Normal Images:   {norm_count}")
                return True

        except Exception as e:
            print(f"⚠️ Error downloading mirror {url}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists("temp_real_dataset"):
                shutil.rmtree("temp_real_dataset")

    return False

if __name__ == "__main__":
    fetch_real_clinical_dataset()
