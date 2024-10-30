import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
from models.diffusion_model import UNet

class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_paths[idx]).convert('RGB')
        target_image = Image.open(self.target_paths[idx]).convert('RGB')
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        return input_image, target_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
train_dataset = ImageDataset(
    "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/denoising",
    "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/original",
    transform=transform,
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model, Loss, Optimizer
model = UNet().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 7
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), f'results/model_epoch_{epoch+1}.pth')
