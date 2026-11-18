import torch
# (You must include your DnCNN class definition here so PyTorch knows the architecture)
from train_dncnn import DnCNN 

# 1. Initialize the blank model structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)

# 2. Load the saved weights into the structure
model.load_state_dict(torch.load("dncnn_sidd_epoch_50.pth"))
model.eval()

# 3. Export to ONNX
dummy_input = torch.randn(1, 3, 256, 256).to(device)
torch.onnx.export(model, dummy_input, "dncnn_sidd_final.onnx", 
                  opset_version=11, 
                  input_names=['input'], 
                  output_names=['output'],
                  dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}})
print("Export rescued and complete!")