# load_realesrgan_model.py

import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def load_realesrgan_model(model_path='models/RealESRGAN_x4plus.pth', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"🔧 Loading Real-ESRGAN model on {device} from {model_path}")

    # Create the model structure
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    # Load into RealESRGAN wrapper
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    print("✅ Real-ESRGAN model loaded successfully.")
    return upsampler
