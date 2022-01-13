# dl_model_convert
Convert(or compile) Deep Learning Models using ONNX, TVM

## Environment
- python3.8

## Install TF-ONNX
```
pip install tensorflow-onnx
```
## Convert TF to ONNX
```
# generating mnist.onnx using saved_model
python -m tf2onnx.convert \
        --saved-model ./output/saved_model \
        --output ./output/mnist1.onnx \
        --opset 7
```
