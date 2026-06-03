Write-Host "Quantizing to INT8..."
python ExportModel/Exporters/exportPTINT8QAT.py

Write-Host "Exporting to ONNX then TensorRT..."
python ExportModel/Exporters/exportINT8TensorRT.py