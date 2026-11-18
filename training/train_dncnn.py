import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1. Paired Dataset for SIDD (with Random Cropping)
class SIDDDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, patch_size=256):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.patch_size = patch_size
        self.image_files = sorted(os.listdir(clean_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load paired images
        clean_img = Image.open(os.path.join(self.clean_dir, img_name)).convert('RGB')
        noisy_img = Image.open(os.path.join(self.noisy_dir, img_name)).convert('RGB')
        
        # Extract dynamic dimensions and calculate a random anchor point
        w, h = clean_img.size
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        
        # Crop identical patches from both images
        clean_crop = clean_img.crop((j, i, j + self.patch_size, i + self.patch_size))
        noisy_crop = noisy_img.crop((j, i, j + self.patch_size, i + self.patch_size))
        
        # Convert to PyTorch Tensors (scales pixels from 0-255 to 0.0-1.0)
        to_tensor = transforms.ToTensor()
        return to_tensor(noisy_crop), to_tensor(clean_crop)

# 2. The DnCNN Architecture
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # The network predicts the noise. Subtract it to return the clean image.
        return x - self.dncnn(x)

# 3. Training Loop and ONNX Export
def train_dncnn(clean_dir, noisy_dir, epochs=50, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    # Initialize Dataset and Dataloader
    dataset = SIDDDataset(clean_dir, noisy_dir, patch_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = DnCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Drops learning rate if the loss plateaus for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(dataloader):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass: Generate denoised image
            denoised_imgs = model(noisy_imgs)
            
            # Calculate Error: Compare denoised image to true clean image
            loss = criterion(denoised_imgs, clean_imgs)
            
            # Backward Pass & Optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}")
        
        epoch_loss = running_loss / len(dataloader)
        scheduler.step(epoch_loss)
        print(f"--- Epoch {epoch+1} Average Loss: {epoch_loss:.6f} ---")
        
        # Save checkpoints safely
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"dncnn_sidd_epoch_{epoch+1}.pth")

    # Final ONNX Export Target
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    torch.onnx.export(model, dummy_input, "dncnn_sidd_final.onnx", 
                      opset_version=11, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}})
    print("Export complete: dncnn_sidd_final.onnx")

if __name__ == "__main__":
    CLEAN_TRAIN_DIR = "./data/sidd/clean"
    NOISY_TRAIN_DIR = "./data/sidd/noisy"
    
    # Create dummy dirs to prevent script crash if run blindly
    os.makedirs(CLEAN_TRAIN_DIR, exist_ok=True)
    os.makedirs(NOISY_TRAIN_DIR, exist_ok=True)
    
    train_dncnn(CLEAN_TRAIN_DIR, NOISY_TRAIN_DIR)