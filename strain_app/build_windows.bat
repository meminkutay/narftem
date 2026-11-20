@echo off
rem Builds dist\nartfem_strains.exe (Windows) - one file, no Python sources inside.
rem Needs Python 3.10+ from python.org (tick "Add python.exe to PATH" when installing).
cd /d "%~dp0"
py -3 -m venv .build_venv || goto :err
call .build_venv\Scripts\activate.bat || goto :err
python -m pip install --upgrade pip || goto :err
python -m pip install numpy "scipy>=1.13" pyinstaller || goto :err
pyinstaller --noconfirm --clean --onefile --name nartfem_strains ^
  --paths . --paths .. ^
  --exclude-module matplotlib --exclude-module tkinter --exclude-module pypardiso --exclude-module pyamg ^
  --exclude-module pandas --exclude-module PIL --exclude-module IPython --exclude-module pytest ^
  --distpath dist --workpath build_tmp --specpath build_tmp app.py || goto :err
echo.
echo Checking the packaged program (runs every example once) ...
dist\nartfem_strains.exe --selftest || goto :err
copy /y HOW_TO_USE.txt dist\ >nul
if not exist dist\examples mkdir dist\examples
copy /y examples\*.json dist\examples\ >nul
echo.
echo Done. Give the student the contents of: %cd%\dist
pause
exit /b 0
:err
echo.
echo The build failed - see the message above.
pause
exit /b 1
