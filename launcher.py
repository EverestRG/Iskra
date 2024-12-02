import win32api, os, sys, time, subprocess

win32api.ShellExecute(0, "open", f"\"{os.path.dirname(sys.argv[0])}\\server.exe\"", "\"1\"", ".", 0)

def run_command():
    # Функция запуска команды
    return subprocess.Popen(["cmd", "/c", "lt --port 5000 --subdomain iskra-ai-server"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

time.sleep(5)

process = run_command()

while True:
    if process.poll() is not None:  # Если процесс завершился
        print("Server down. Restarting...")

        stdout, stderr = process.communicate()
        print("Вывод команды:", stdout)
        print("Ошибки:", stderr)

        # Перезапускаем процесс
        process = run_command()
    else:
        # Если процесс всё ещё работает, подождём немного
        time.sleep(1)