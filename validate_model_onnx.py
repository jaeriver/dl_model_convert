import numpy as np
import onnxruntime as ort
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='mobilenet-13.onnx' , type=str)
args = parser.parse_args()
onnx_model = args.model

img = np.array((1,3,640,640)).astype(np.float32)
  
session = ort.InferenceSession(onnx_model)

session.get_modelmeta()
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]
print(inname, outname)

res = session.run(outname, {inname[0]: img})
print("the expected result is \"7\"")
print("the digit is classified as \"%s\" in ONNXRruntime"%np.argmax(res))
