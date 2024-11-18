from torch.quantization import quantize_dynamic
from models.diffusion_model import UNet

model = UNet()
quantized_model = quantize_dynamic(model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8)
torch.save(quantized_model.state_dict(), 'results/quantized_model.pth')
