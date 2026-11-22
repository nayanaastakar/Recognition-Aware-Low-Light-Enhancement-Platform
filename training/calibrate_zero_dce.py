"""
Optimizes Zero-DCE non-reference loss on low-light real samples to produce calibrated dynamic curves.
"""
import torch
import torch.optim as optim
from PIL import Image
from models.zero_dce.model import DCE_Net, L_spatial, L_exposure, L_color, L_tv
from models.transforms import to_tensor
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dark_img_path = 'C:/Users/HP/.gemini/antigravity/brain/02143033-8a71-4cf2-b23c-d6fe4a2d8670/.user_uploaded/media_1795282431807_bc3c6c33.jpg'
img = Image.open(dark_img_path).convert('RGB')
x = to_tensor(img).unsqueeze(0)

model = DCE_Net()
opt = optim.Adam(model.parameters(), lr=1e-3)
spa = L_spatial()
exp = L_exposure()
col = L_color()
tv = L_tv()

print("[*] Calibrating Zero-DCE non-reference tone curves...")
for step in range(25):
    opt.zero_grad()
    enh, A = model(x)
    loss = spa(x, enh) + 10.0 * exp(enh) + 5.0 * col(enh) + 200.0 * tv(A)
    loss.backward()
    opt.step()
    if (step + 1) % 5 == 0:
        print(f"  Step {step+1:02d}: Enhanced Mean={enh.mean().item():.3f}, A Mean={A.mean().item():.3f}")

out_ckpt = os.path.join(root, 'checkpoints', 'zero_dce_model.pth')
torch.save({'state_dict': model.state_dict()}, out_ckpt)
print(f"[+] Saved calibrated Zero-DCE checkpoint to {out_ckpt}")
