import onnxruntime as ort
import numpy as np

# load ONNX
session = ort.InferenceSession("../model.onnx")

# new input array
new_mfcc = [
    0.217108, -0.725244, -0.491752, 0.283354, 0.497911, 0.041176,
    -0.357519, -0.234177, 0.163444, 0.280876, 0.013326, -0.236523,
    -0.11199, 0.144984, 0.161907, -0.0648197, -0.15113, 0.00616863,
    0.109725
]

# wrap in 2D array and set dtype
x_input = np.array([new_mfcc], dtype=np.float32)

# get input name and run model
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: x_input})
print(outputs)
