import zipfile

zip_path = r"C:\Users\HP\Downloads\ExDark.zip"
print(f"Opening {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    print(f"Total files in ExDark.zip: {len(names)}")
    matches = [n for n in names if "2015_00026" in n]
    print("Matches for 2015_00026 in ExDark.zip:", matches)
    if matches:
        out_path = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement\data\samples\low\2015_00026_original.jpg"
        with open(out_path, "wb") as f:
            f.write(z.read(matches[0]))
        print(f"Extracted original image to: {out_path}")
