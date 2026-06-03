Write-Host "Quantizing to INT8..."
python Exporters/exportPTINT8QAT.py

Write-Host "Exporting to ONNX then TensorRT..."
python Exporters/exportINT8TensorRT.py