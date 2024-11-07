import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from models.diffusion_model import UNet

model = UNet().cuda()
model.load_state_dict(torch.load('results/model_epoch_10.pth'))
model.eval()

test_dataset = ImageDataset(
    "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/denoising",
    "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/original",
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

psnr_list = []
ssim_list = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.cuda()
        outputs = model(inputs).cpu().numpy().squeeze().transpose(1, 2, 0)
        targets = targets.numpy().squeeze().transpose(1, 2, 0)
        outputs = np.clip(outputs, 0, 1)

        psnr_value = peak_signal_noise_ratio(targets, outputs, data_range=1)
        ssim_value = structural_similarity(targets, outputs, multichannel=True, data_range=1)

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

print(f"Average PSNR: {np.mean(psnr_list):.2f}")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
