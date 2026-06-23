import onnxruntime as ort
import numpy as np

MODEL_PATH = "models/anti-spoof/minifasnet_v2.onnx"

session = ort.InferenceSession(MODEL_PATH)

print("INPUTS:")
for i in session.get_inputs():
    print(i.name, i.shape, i.type)

print("\nOUTPUTS:")
for o in session.get_outputs():
    print(o.name, o.shape, o.type)