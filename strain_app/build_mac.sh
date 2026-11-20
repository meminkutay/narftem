#!/bin/bash
# Builds dist/nartfem_strains (macOS, or Linux) - one file, no Python sources inside.
# Run on the same kind of computer the student uses (Apple-silicon build runs on Apple-silicon Macs).
set -e
cd "$(dirname "$0")"
python3 -m venv .build_venv
source .build_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy "scipy>=1.13" pyinstaller
pyinstaller --noconfirm --clean --onefile --name nartfem_strains \
  --paths . --paths .. \
  --exclude-module matplotlib --exclude-module tkinter --exclude-module pypardiso --exclude-module pyamg \
  --exclude-module pandas --exclude-module PIL --exclude-module IPython --exclude-module pytest \
  --distpath dist --workpath build_tmp --specpath build_tmp app.py
echo
echo "Checking the packaged program (runs every example once) ..."
./dist/nartfem_strains --selftest
cp HOW_TO_USE.txt dist/
mkdir -p dist/examples && cp examples/*.json dist/examples/
echo
echo "Done. Give the student the contents of: $(pwd)/dist"
