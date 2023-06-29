rmdir build /s
if errorlevel 1 exit 1
conda build --output-folder build .
if errorlevel 1 exit 1