import torch
import cv2
import numpy as np
from SwinIR.models.network_swinir import SwinIR


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once
model = SwinIR(
    upscale=2,
    in_chans=3,
    img_size=64,
    window_size=8,
    img_range=1.,
    depths=[6,6,6,6,6,6],
    embed_dim=60,
    num_heads=[6,6,6,6,6,6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
)

model.load_state_dict(
    torch.load("models/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth")["params"]
)
model.eval()
model = model.to(device)


def enhance_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.

    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze().permute(1,2,0).cpu().numpy()
    output = (output * 255.0).clip(0,255).astype(np.uint8)

    out_path = "enhanced.png"
    cv2.imwrite(out_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    return out_path
