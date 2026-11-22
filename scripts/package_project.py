import os
import zipfile

def package():
    base_dir = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement"
    zip_path = os.path.join(base_dir, "Recognition_Aware_Low_Light_Image_Enhancement_Final.zip")
    
    # Files and folders to exclude
    exclude_dirs = {"__pycache__", ".git", ".pytest_cache", ".streamlit", "official_dfine", "official_zero_dce", "official_retinexformer", "official_uretinex_net", "official_retinexnet"}
    exclude_exts = {".pyc", ".pyo"}
    
    print(f"Creating package: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith(".zip") or any(file.endswith(ext) for ext in exclude_exts):
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)
                z.write(file_path, rel_path)
    
    print(f"Package successfully created: {zip_path} ({os.path.getsize(zip_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    package()
