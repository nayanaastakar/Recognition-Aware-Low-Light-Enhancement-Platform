from PIL import Image
import numpy as np

images = [
    ('Original Bicycle Scene', 'C:/Users/HP/.gemini/antigravity/scratch/Recognition_Aware_Low_Light_Image_Enhancement/data/samples/low/bicycle_extracted.png'),
    ('Zero-DCE Output', 'C:/Users/HP/.gemini/antigravity/scratch/Recognition_Aware_Low_Light_Image_Enhancement/results/figures/test_dce_bicycle.png'),
    ('IGR-Net (Proposed)', 'C:/Users/HP/.gemini/antigravity/scratch/Recognition_Aware_Low_Light_Image_Enhancement/results/figures/test_igr_bicycle.png'),
    ('Retinexformer (Adapted)', 'C:/Users/HP/.gemini/antigravity/scratch/Recognition_Aware_Low_Light_Image_Enhancement/results/figures/test_ret_bicycle.png')
]

for name, path in images:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    print(f"=== {name} ===")
    print(f"  Path: {path}")
    print(f"  Resolution: {img.width}x{img.height}")
    print(f"  Pixel Range: [{int(arr.min())}, {int(arr.max())}]")
    print(f"  Mean Intensity: {arr.mean():.2f}")
    print(f"  Std Dev (Contrast): {arr.std():.2f}")
    print(f"  RGB Means: R={arr[:,:,0].mean():.2f}, G={arr[:,:,1].mean():.2f}, B={arr[:,:,2].mean():.2f}")
    print()
