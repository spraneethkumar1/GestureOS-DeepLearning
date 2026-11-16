@echo off
setlocal
echo === GestureOS setup ===
py -3.12 --version
if errorlevel 1 (
  echo Python 3.12 was not found. Install Python 3.12 and enable "Add Python to PATH".
  pause
  exit /b 1
)
if not exist .venv (
  py -3.12 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package installation failed. Check your internet connection and the error above.
  pause
  exit /b 1
)
echo.
echo Setup complete. Next run: collect_data.bat
pause
