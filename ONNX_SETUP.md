# ONNX Face Recognition

Enhanced accuracy using ONNX models (optional).

## Installation

```bash
pip install onnxruntime>=1.16.0
```

## Benefits

- Better accuracy than OpenCV
- CPU optimized 
- Auto fallback to OpenCV
- Models download automatically

## Disable ONNX

```python
fr = SimpleFaceRecognition(use_onnx=False)
```