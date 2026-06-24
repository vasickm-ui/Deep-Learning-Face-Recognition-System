import onnxruntime as ort
import numpy as np


MINIFAS_PATH = "models/anti-spoof/minifasnet_v2.onnx"
MAX_SPOOF_15_PATH = "models/anti-spoof/AntiSpoofing_bin_15_128.onnx"
MAX_SPOOF_PRINT_REPLAY_15_128 = "models/anti-spoof/AntiSpoofing_print-replay_15_128.onnx"

session = ort.InferenceSession(MAX_SPOOF_PRINT_REPLAY_15_128)

inp = session.get_inputs()[0]
print("\nINPUT")
print("name:", inp.name)
print("shape:", inp.shape)
print("type:", inp.type)


out = session.get_outputs()[0]
print("\nOUTPUT")
print("name:", out.name)
print("shape:", out.shape)
print("type:", out.type)


shape = inp.shape
shape = [1 if isinstance(x, str) else x for x in shape]

dummy = np.random.rand(*shape).astype(np.float32)

print("\nDummy input shape:", dummy.shape)

result = session.run(None, {inp.name: dummy})

print("\nRAW OUTPUT")
print(result)

print("\nINTERPRETATION HINT")

r = result[0]
print("output array shape:", r.shape)

print("min value:", np.min(r))
print("max value:", np.max(r))
print("mean value:", np.mean(r))