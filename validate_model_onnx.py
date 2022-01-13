import numpy as np
import onnxruntime as ort

onnx_model = 'mobilenet.onnx'
img = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=numpy.float32)
  
sess_ort = ort.InferenceSession(onnx_model)

sess_ort.get_modelmeta()
first_input_name = sess_ort.get_inputs()[0].name
first_output_name = sess_ort.get_outputs()[0].name

print(first_input_name, first_output_name)

res = sess_ort.run(None, {first_input_name: img})
print("the expected result is \"7\"")
print("the digit is classified as \"%s\" in ONNXRruntime"%np.argmax(res))
