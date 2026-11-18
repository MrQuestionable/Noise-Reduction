import torch
import torch.nn as nn

# 1. Define a lightweight DnCNN
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
        # DnCNN predicts the noise; subtract it from original to get clean image
        return x - self.dncnn(x)

model = DnCNN()
# ... [Insert standard PyTorch training loop here using your synthetic/SIDD data] ...

# 2. Export the trained model to ONNX
dummy_input = torch.randn(1, 3, 256, 256) # Batch=1, Channels=3, 256x256 image
torch.onnx.export(model, dummy_input, "dncnn_denoiser.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}})