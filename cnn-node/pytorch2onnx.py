import torch
from _network import CNNODE,CNN_LSTM

# chech pytorch version
print(torch.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNNODE()
model.load_state_dict(torch.load('torch_models/saved_model_ODE.pth'))
model.eval()

dummy_input = torch.randn(1,3, 672, 224)
torch.onnx.export(model, dummy_input, 'torch_models/model_ODE.onnx', verbose=True, export_params=True, opset_version=11, input_names=['input'], output_names=['output'])

