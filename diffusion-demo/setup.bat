@echo off
echo ========================================
echo  Instalando dependencias del proyecto
echo ========================================

py -m venv venv
call venv\Scripts\activate

echo Instalando PyTorch con soporte CUDA...
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo Instalando resto de dependencias...
py -m pip install -r requirements.txt

echo.
echo ========================================
echo  Listo! Ahora crea un archivo .env con:
echo  HF_TOKEN=tu_token_de_huggingface
echo ========================================
echo.
pause