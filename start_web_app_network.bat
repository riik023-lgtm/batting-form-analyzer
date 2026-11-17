@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$ip=(Get-NetIPAddress -AddressFamily IPv4 ^| Where-Object {$_.InterfaceAlias -match 'Wi-Fi' -and $_.IPAddress -notlike '169.254*'} ^| Select-Object -First 1 -ExpandProperty IPAddress); if($ip){$ip}"`) do set "LOCAL_IP=%%I"
echo ==============================================================
echo Smartphone / same-Wi-Fi mode
if defined LOCAL_IP (
  echo スマホで開くURL: http://%LOCAL_IP%:8502
) else (
  echo IPv4を自動取得できませんでした。別画面で ipconfig を実行してください。
)
echo この画面はアプリ使用中は閉じないでください。
echo ==============================================================
"%PY%" -m streamlit run app.py --server.address 0.0.0.0 --server.port 8502
