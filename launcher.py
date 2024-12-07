import os, sys, time, subprocess

def run_command():
    # Функция запуска команды
    return subprocess.Popen(["cmd", "/c", "lt --port 5000 --subdomain iskra-ai-server"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            shell=True,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

def start_server():
    return subprocess.Popen([f"{os.path.dirname(sys.argv[0])}\\server.exe", "1"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            shell=True,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

print('Starting server...')
serverprocess = start_server()

time.sleep(5)
print("Server started!")

print('Starting listener...')
process = run_command()
print('Listener started!')

while True:
    if process.poll() is not None:  # Если процесс завершился
        print("Listener down. Restarting...")

        stdout, stderr = process.communicate()
        print("Listener output:", stdout)
        print("Listener error:", stderr)

        # Перезапускаем процесс
        process = run_command()
    else:
        # Если процесс всё ещё работает, подождём немного
        time.sleep(1)