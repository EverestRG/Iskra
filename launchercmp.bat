python -m PyInstaller --noconfirm --onefile --console --name "Launcher" --icon "icon.ico"  launcher.py
move .\dist\Launcher.exe .\dist\server\
rd /s/q .\build
del /Q .\Launcher.spec
pause