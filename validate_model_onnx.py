import numpy as np
import onnxruntime as ort
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='mobilenet-13.onnx' , type=str)
parser.add_argument('--batch_size',default=1 , type=int)
args = parser.parse_args()
batch_size = args.batch_size
onnx_model = args.model

def make_dataset(batch_size,size):
    image_shape = (size, size,3)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

data, image_shape = make_dataset(batch_size, 224)
  
session = ort.InferenceSession(onnx_model)

session.get_modelmeta()
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]
print(inname, outname)

start_time = time.time()
res = session.run(outname, {inname[0]: data})
print('batch_size: ', batch_size,'time: ', time.time() - start_time)
print(res)
print("the digit is classified as \"%s\" in ONNXRruntime"%np.argmax(res))
