python -m PyInstaller --noconfirm --onedir --console --name "server" --icon "icon.ico"  server.py
python -m PyInstaller --noconfirm --onefile --console --name "Launcher" --icon "icon.ico"  launcher.py
move .\dist\Launcher.exe .\dist\server\
rd /s/q .\build
del /Q .\server.spec
del /Q .\Launcher.spec
pause