# requirements.ps1

$ErrorActionPreference = "Stop"

Write-Host "Upgrading pip..."
python -m ensurepip --upgrade
python -m pip install --upgrade pip

Write-Host "Installing HuggingFace Hub..."
python -m pip install -U huggingface_hub

Write-Host "Installing ultralytics..."
python -m pip install -U ultralytics

Write-Host "Installing Albumentations..."
python -m pip install -U albumentationsx

Write-Host "Installing TensorRT (CUDA 13)..."
python -m pip install --upgrade tensorrt-cu13

Write-Host "Installing ONNX..."
python -m pip install -U onnx

Write-Host "Installing NVIDIA ModelOpt (ONNX)..."
python -m pip install -U "nvidia-modelopt[onnx]"
python -m pip uninstall -y cupy-cuda12x onnxruntime-genai-cuda
python -m pip install -U cupy-cuda13x
python -m pip install --pre -U onnxruntime-genai-cuda

Write-Host "Installing pytest..."
python -m pip install -U pytest

Write-Host "Installing colored..."
python -m pip install -U colored

Write-Host "Ensuring OpenCV installation success..."
python -m pip uninstall opencv-python-headless opencv-python -y
python -m pip install -U opencv-python

Write-Host "Upgrading all packages..."
python -m pip freeze > r.txt
((Get-Content r.txt) -replace '==', '>=') | Set-Content r.txt
python -m pip install --upgrade -r r.txt
Remove-Item r.txt

Write-Host "Uninstalling torch, torchvision, onnxruntime-gpu..."
python -m pip uninstall -y torch torchvision onnxruntime-gpu

Write-Host "Installing PyTorch (CUDA 13 index)..."
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132

Write-Host "Installing onnxruntime-gpu..."
python -m pip install -U coloredlogs flatbuffers numpy packaging protobuf sympy
python -m pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ --upgrade onnxruntime-gpu

Write-Host "Finished installing requirements"
