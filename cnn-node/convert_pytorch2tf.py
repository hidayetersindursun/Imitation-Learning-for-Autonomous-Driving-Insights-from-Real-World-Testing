import torch
import torchvision
import onnx
import onnx_tf
import tensorflow as tf
import numpy as np
from _network import CNN_LSTM, CNNODE
from onnx_tf.backend import prepare

# Load the saved PyTorch model
#pytorch_model = CNN_LSTM()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_model = CNNODE().to(device)
pytorch_model.load_state_dict(torch.load('torch_models/saved_model_ODE.pth'))   

# Create an example input tensor
example_input = torch.randn(1, 3, 672, 224).to(device)  # Example input tensor

# Export the PyTorch model to ONNX format
torch.onnx.export(pytorch_model, example_input, 'model_ODE.onnx', export_params=True)


# Load the ONNX model into TensorFlow
#onnx_model = onnx.load('model_ODE.onnx')
#prepare(onnx_model).export_graph('model')
