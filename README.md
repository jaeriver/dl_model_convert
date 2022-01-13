# dl_model_convert
Convert(or compile) Deep Learning Models using ONNX, TVM

## Environment
- python3.8

## Install TF-ONNX
```
pip install -U tf2onnx
```
## Convert TF to ONNX
```
# generating mnist.onnx using saved_model
python -m tf2onnx.convert \
        --saved-model ./tf_saved_model \
        --output ./converted_model.onnx \
        --opset 7
```

## Validate converted ONNX model
```
pip install onnxruntime
```
